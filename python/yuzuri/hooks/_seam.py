"""wrap(): replace a function or method at a boundary, after checking it is the one the hook was written against."""
import functools
import inspect


class SeamError(RuntimeError):
    pass


def wrap(owner, attr: str, params: list, make, adds: tuple = (), needs: tuple = ()) -> None:
    """owner.attr becomes make(original). The original's parameter names must be params, in order: SGLang changed
    the function otherwise, and the launch stops with what it found (the import of owner's module fails). A last
    entry "..." checks the leading names only: the hook passes the rest through untouched. adds names what the hook
    brings (module or attribute names): an original that already mentions one is a tree patched with the same
    mechanism, and running both would do the work twice, so the launch stops there too. needs names what the hook
    relies on inside the original (an expression it corrects from outside): without it the launch stops as well."""
    where = ".".join(n for n in (getattr(owner, "__module__", ""),
                                 getattr(owner, "__qualname__", getattr(owner, "__name__", str(owner)))) if n)
    raw = inspect.getattr_static(owner, attr, None)
    if raw is None:
        raise SeamError(f"yuzuri: {where} has no {attr}: this SGLang is not the one the hook was written for")
    orig = getattr(owner, attr)
    got = list(inspect.signature(raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw).parameters)
    params = list(params)
    if params and params[-1] == "...":
        params = params[:-1]
        got = got[: len(params)]
    if got != params:
        raise SeamError(f"yuzuri: {where}.{attr} takes {got}, the hook was written for {params}")
    if adds or needs:
        try:
            src = inspect.getsource(raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw)
        except (OSError, TypeError):
            src = ""
        gone = [t for t in needs if t not in src]
        if gone:
            raise SeamError(f"yuzuri: {where}.{attr} no longer has {gone}: the hook corrects that expression from "
                            f"outside and has nothing to attach to in this SGLang")
        have = [t for t in adds if t in src]
        if have:
            raise SeamError(f"yuzuri: {where}.{attr} already uses {have}: this SGLang carries a patch with the same "
                            f"mechanism (the DeepSeek-V4.1 recipe's tree?). Yuzuri goes on an SGLang without it")
    new = make(raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else orig)
    functools.update_wrapper(new, raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw)
    if isinstance(raw, staticmethod):
        new = staticmethod(new)
    elif isinstance(raw, classmethod):
        new = classmethod(new)
    setattr(owner, attr, new)


def _compile_into(module_globals: dict, source: str, label: str, flags: int = 0) -> dict:
    """Run source with module_globals as its globals; inspect and tracebacks find the text under label."""
    import linecache

    name = f"<yuzuri: {label}>"
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    ns = {}
    exec(compile(source, name, "exec", flags=flags, dont_inherit=True), module_globals, ns)
    return ns


def define(module, source: str, label: str) -> None:
    """The functions source defines become module's own (a kernel helper that the module's kernels call by name).
    A name the module already has stops the launch."""
    have = [n for n in _compile_into(dict(vars(module)), source, label + " (names)") if hasattr(module, n)]
    if have:
        raise SeamError(f"yuzuri: {module.__name__} already has {have}: this SGLang carries a patch with the same "
                        f"mechanism. Yuzuri goes on an SGLang without it")
    vars(module).update(_compile_into(vars(module), source, label))


def rewrite(owner, attr: str, edits: list, adds: tuple = ()) -> None:
    """owner.attr is compiled again from its own source with each (old, new) of edits applied: a change in the
    middle of a function, where calling the original before or after cannot make it. Each old must be in the source
    exactly once, or the launch stops with the one that is not: SGLang (or a patch on it) changed that line. adds as
    in wrap(). A Triton kernel is made again from the new source; behind any other decorator the function takes the
    new body in place. The new text sees the module's
    globals and nothing else: what it needs it imports itself."""
    import __future__
    import textwrap

    where = ".".join(n for n in (getattr(owner, "__module__", ""),
                                 getattr(owner, "__qualname__", getattr(owner, "__name__", str(owner)))) if n)
    raw = inspect.getattr_static(owner, attr, None)
    if raw is None:
        raise SeamError(f"yuzuri: {where} has no {attr}: this SGLang is not the one the hook was written for")
    kind = type(raw) if isinstance(raw, (staticmethod, classmethod)) else None
    fn = raw.__func__ if kind else raw
    fn = getattr(fn, "fn", fn)  # a Triton JITFunction keeps the Python function it was made from
    jit = fn is not (raw.__func__ if kind else raw)
    fn = inspect.unwrap(fn)
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        raise SeamError(f"yuzuri: the source of {where}.{attr} cannot be read: the hook changes a line inside it")
    if getattr(fn, "__closure__", None) and any(n != "__class__" for n in fn.__code__.co_freevars):
        raise SeamError(f"yuzuri: {where}.{attr} closes over {list(fn.__code__.co_freevars)}: it cannot be compiled "
                        f"again on its own")
    have = [t for t in adds if t in src]
    if have:
        raise SeamError(f"yuzuri: {where}.{attr} already uses {have}: this SGLang carries a patch with the same "
                        f"mechanism (the DeepSeek-V4.1 recipe's tree?). Yuzuri goes on an SGLang without it")
    for old, new in edits:
        if src.count(old) != 1:
            raise SeamError(f"yuzuri: {where}.{attr} has {src.count(old)} of {old!r}, the hook was written for one: "
                            f"this SGLang changed the line the hook rewrites")
        src = src.replace(old, new)
    flags = fn.__code__.co_flags & __future__.annotations.compiler_flag
    if inspect.isclass(owner):
        # a method: zero-argument super() finds its class the way it does in a class body
        body = "def __yuzuri_method(__class__):\n" + textwrap.indent(src, "    ") + f"\n    return {fn.__name__}\n"
        made = _compile_into(fn.__globals__, body, f"{where}.{attr}", flags)["__yuzuri_method"](owner)
        if isinstance(made, (staticmethod, classmethod)):
            made = made.__func__
        made.__qualname__ = fn.__qualname__
    elif not jit and inspect.getattr_static(owner, attr) is not fn:
        # what the module holds was made from the function by a decorator that is not to run twice (a torch op is
        # registered once, and the registration keeps the function): the function itself takes the new body, and
        # everything made from it runs that
        lines = src.splitlines(True)
        body = "".join(lines[next(i for i, l in enumerate(lines) if l.startswith(("def ", "async def "))):])
        made = _compile_into(fn.__globals__, body, f"{where}.{attr}", flags)[fn.__name__]
        if made.__code__.co_freevars or made.__defaults__ != fn.__defaults__ or made.__kwdefaults__ != fn.__kwdefaults__:
            raise SeamError(f"yuzuri: {where}.{attr} cannot take a new body in place (its defaults or free variables "
                            f"would change)")
        fn.__code__ = made.__code__
        return
    else:
        made = _compile_into(fn.__globals__, src, f"{where}.{attr}", flags)[fn.__name__]
    setattr(owner, attr, kind(made) if kind else made)
