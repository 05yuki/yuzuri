"""wrap(): replace a function or method at a boundary, after checking it is the one the hook was written against."""
import functools
import inspect


class SeamError(RuntimeError):
    pass


_CARRIED = set()
_RUNS = []  # the hooks being put in, innermost last: [name, what it changed, seams tried, seams the tree had]


def begin(name: str) -> list:
    _RUNS.append([name, [], 0, 0])
    return _RUNS[-1]


def end(mark: list) -> None:
    if mark in _RUNS:
        _RUNS.remove(mark)


def undo(mark: list) -> None:
    """Take back what the hook put in before it failed, last first."""
    for owner, attr, old in reversed(mark[1]):
        if old is _ABSENT:
            if isinstance(owner, dict):
                owner.pop(attr, None)
            else:
                delattr(owner, attr)
        elif isinstance(owner, dict):
            owner[attr] = old
        else:
            setattr(owner, attr, old)
    mark[1].clear()


def carried_since(mark: list) -> str:
    """"all" when every seam the hook tried was the tree's own, "some", or ""."""
    return "" if not mark[3] else "all" if mark[3] == mark[2] else "some"


_ABSENT = object()


def _put(owner, attr: str, new) -> None:
    """setattr that a failed optional hook can take back."""
    if _RUNS:
        _RUNS[-1][1].append((owner, attr, inspect.getattr_static(owner, attr, _ABSENT)))
    setattr(owner, attr, new)


def _tried() -> None:
    if _RUNS:
        _RUNS[-1][2] += 1


def _carried(where: str, have: list) -> None:
    """The tree already has what the hook brings (a patch with the same mechanism): the tree's own is the one that
    runs and the hook stays off, said once a process and place."""
    if _RUNS:
        _RUNS[-1][3] += 1
        from yuzuri import _features

        _features.note(_RUNS[-1][0], "tree", f"{where} has {have}")
    if where not in _CARRIED:
        _CARRIED.add(where)
        print(f"[yuzuri] {where} already has {have}: this SGLang carries the same mechanism, its own runs and the "
              f"hook stays off", flush=True)


def _by_name(sig, params: list, make, target):
    """The hook's function (written for params, in that order) on an original whose parameters are sig: a call is
    bound by name, the hook gets its own parameters, and when it calls the original the caller's other arguments
    go back in where they were."""
    import threading

    held = threading.local()
    bound_self = len(inspect.signature(target).parameters) < len(sig.parameters)  # a bound method: no self here
    names = [p for p in params if not (bound_self and p == next(iter(sig.parameters)))]
    tsig = inspect.signature(target)

    def original(*a, **k):
        mine = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                    default=tsig.parameters[n].default) for n in names]).bind(*a, **k)
        ba = tsig.bind_partial()
        ba.arguments.update(held.stack[-1])
        ba.arguments.update(mine.arguments)
        ordered = {n: ba.arguments[n] for n in tsig.parameters if n in ba.arguments}
        ba.arguments.clear()
        ba.arguments.update(ordered)
        return target(*ba.args, **ba.kwargs)

    hook = make(original)

    def adapter(*a, **k):
        ba = tsig.bind(*a, **k)
        rest = {n: v for n, v in ba.arguments.items() if n not in names}
        ba.apply_defaults()
        if not hasattr(held, "stack"):
            held.stack = []
        held.stack.append(rest)
        try:
            return hook(*[ba.arguments[n] for n in names])
        finally:
            held.stack.pop()

    return adapter


def wrap(owner, attr: str, params: list, make, adds: tuple = (), needs: tuple = ()) -> None:
    """owner.attr becomes make(original). The original's parameter names are params, in order. When SGLang added or moved
    parameters and every name of params is still there, the hook is bound by name and the others pass through; when
    one of params is gone the launch stops with what it found (the import of owner's module fails). A last
    entry "..." checks the leading names only: the hook passes the rest through untouched. adds names what the hook
    brings (module or attribute names): an original that already mentions one is a tree patched with the same
    mechanism: the tree's own runs and the hook stays off (running both would do the work twice). needs names what the hook
    relies on inside the original (an expression it corrects from outside): without it the launch stops as well."""
    _tried()
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
    adapt = None
    if got != params:
        # the names the hook works with are all still there: what SGLang added or moved passes through untouched
        fn0 = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        sig = inspect.signature(fn0)
        kinds = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        gone = [p for p in params if p not in sig.parameters or sig.parameters[p].kind in kinds]
        if gone or inspect.iscoroutinefunction(fn0) or inspect.isgeneratorfunction(fn0) \
                or inspect.isasyncgenfunction(fn0):
            raise SeamError(f"yuzuri: {where}.{attr} takes {list(sig.parameters)}, the hook was written for {params}"
                            + (f" and {gone} is gone" if gone else ""))
        adapt = sig
        print(f"[yuzuri] {where}.{attr} takes {list(sig.parameters)} where the hook was written for {params}: the "
              f"hook works with its own names, the others pass through", flush=True)
    if adds or needs:
        try:
            src = inspect.getsource(raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw)
        except (OSError, TypeError):
            src = ""
        # the tree's own mechanism is looked for first: a tree that carries it has rewritten the expressions the hook
        # would have relied on, and that is not a missing seam
        have = [t for t in adds if t in src]
        if have:
            _carried(f"{where}.{attr}", have)
            return
        gone = [t for t in needs if t not in src]
        if gone:
            raise SeamError(f"yuzuri: {where}.{attr} no longer has {gone}: the hook corrects that expression from "
                            f"outside and has nothing to attach to in this SGLang")
    if adapt is not None:
        new = _by_name(adapt, params, make, raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else orig)
    else:
        new = make(raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else orig)
    functools.update_wrapper(new, raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw)
    if isinstance(raw, staticmethod):
        new = staticmethod(new)
    elif isinstance(raw, classmethod):
        new = classmethod(new)
    _put(owner, attr, new)


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
    A name the module already has is a tree with the same mechanism: its own stay."""
    _tried()
    have = [n for n in _compile_into(dict(vars(module)), source, label + " (names)") if hasattr(module, n)]
    if have:
        _carried(module.__name__, have)
        return
    for n, v in _compile_into(dict(vars(module)), source, label).items():
        _put(module, n, v)


def _spaced(old: str):
    """old as a pattern that takes any run of spaces or line breaks where old has one, the indentation in front and
    the line end behind included."""
    import re

    words = old.split()
    head = r"[ \t]*" if old[:1].isspace() else ""
    tail = r"[ \t]*\n" if old.endswith("\n") else ""
    return re.compile(head + r"\s+".join(re.escape(w) for w in words) + tail)


def rewrite(owner, attr: str, edits: list, adds: tuple = ()) -> None:
    """owner.attr is compiled again from its own source with each (old, new) of edits applied: a change in the
    middle of a function, where calling the original before or after cannot make it. Each old must be in the source
    exactly once, or the launch stops with the one that is not: SGLang (or a patch on it) changed that line. adds as
    in wrap() (the tree's own runs, nothing is rewritten). A Triton kernel is made again from the new source; behind any other decorator the function takes the
    new body in place. The new text sees the module's
    globals and nothing else: what it needs it imports itself."""
    import __future__
    import textwrap

    _tried()
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
        _carried(f"{where}.{attr}", have)
        return
    for old, new in edits:
        if src.count(old) == 1:
            src = src.replace(old, new)
            continue
        # the same words spaced or wrapped another way (a formatter ran over the file): still the line the hook means
        found = _spaced(old).findall(src) if src.count(old) == 0 else []
        if len(found) != 1:
            raise SeamError(f"yuzuri: {where}.{attr} has {src.count(old)} of {old!r}, the hook was written for one: "
                            f"this SGLang changed the line the hook rewrites")
        print(f"[yuzuri] {where}.{attr}: a line the hook rewrites is spaced differently in this SGLang, matched by "
              f"its words", flush=True)
        src = src.replace(found[0], new)
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
        _put(fn, "__code__", made.__code__)
        return
    else:
        made = _compile_into(fn.__globals__, src, f"{where}.{attr}", flags)[fn.__name__]
    _put(owner, attr, kind(made) if kind else made)
