"""YUZURI_PARK=1: the server can leave the cards and come back (kt_park): after the load the model's tensors are copied
into a region that can be paused with a host copy, the scheduler of every rank gets yuzuri_park / yuzuri_resume (called on all ranks by
SGLang's collective RPC), the HTTP server gets POST /yuzuri/park and /yuzuri/resume and GET /yuzuri/state,
and answers 503 to any other POST while parked. tools/front.py puts several such servers behind one address."""
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError, wrap

REPLACES = ["srt/model_executor/model_runner.py", "srt/managers/scheduler.py", "srt/entrypoints/http_server.py",
            "srt/runtime_context.py"]
ON = os.environ.get("YUZURI_PARK") == "1"


def _runner(m):
    if not ON:
        return

    def load_model(orig):
        def load_model(self):
            from sglang.srt.layers.moe import kt_park

            got = orig(self)
            kt_park.adopt(self.model)
            return got

        return load_model

    wrap(m.ModelRunner, "load_model", ["self"], load_model)


def _scheduler(m):
    if not ON:
        return
    S = getattr(m, "Scheduler", None)
    if S is None or not hasattr(S, "handle_rpc_request"):
        raise SeamError(f"yuzuri: {m.__name__} has no Scheduler.handle_rpc_request: nothing carries a park to every rank")

    def _busy(self) -> int:
        n = 0
        for name in ("waiting_queue", "grammar_queue"):
            n += len(getattr(self, name, None) or ())
        batch = getattr(self, "running_batch", None)
        n += len(getattr(batch, "reqs", None) or ())
        return n

    def yuzuri_park(self):
        from sglang.srt.layers.moe import kt_park

        if _busy(self):
            raise RuntimeError("requests are running or queued: a server parks when it is idle")
        kt_park.park()

    def yuzuri_resume(self):
        from sglang.srt.layers.moe import kt_park

        kt_park.resume()

    S.yuzuri_park, S.yuzuri_resume = yuzuri_park, yuzuri_resume


def _http(m):
    if not ON:
        return
    app = getattr(m, "app", None)
    if app is None or not hasattr(m, "_global_state"):
        raise SeamError(f"yuzuri: {m.__name__} has no app or _global_state: the park has no route")
    from sglang.srt.managers.io_struct import RpcReqInput

    from fastapi.responses import JSONResponse

    state = {"parked": False}

    async def _call(method, parked):
        # the door closes before the park starts: a request that slips in would find the weights gone
        was, state["parked"] = state["parked"], state["parked"] or parked
        out = await m._global_state.tokenizer_manager.collective_rpc(RpcReqInput(method=method))
        state["parked"] = parked if out.success else was
        return {"success": bool(out.success), "message": out.message, "parked": state["parked"]}

    class _Door:
        """Plain ASGI, outside the routes: a parked server answers 503 to every POST but its own."""

        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if (scope["type"] == "http" and state["parked"] and scope["method"] == "POST"
                    and not scope["path"].startswith("/yuzuri/")):
                answer = JSONResponse({"error": {"message": "this server is parked: POST /yuzuri/resume first",
                                                 "type": "yuzuri_parked", "code": 503}}, status_code=503)
                return await answer(scope, receive, send)
            return await self.inner(scope, receive, send)

    app.add_middleware(_Door)

    @app.post("/yuzuri/park")
    async def yuzuri_park():
        return await _call("yuzuri_park", True)

    @app.post("/yuzuri/resume")
    async def yuzuri_resume():
        return await _call("yuzuri_resume", False)

    @app.get("/yuzuri/state")
    async def yuzuri_state():
        return {"parked": state["parked"]}


def _buffers(m):
    """SGLang's named persistent buffers (runtime_context.get_buffer: the attention backend's workspace) are made in
    the park's region: they leave the card with it and come back at the same address."""
    if not ON:
        return
    ctx = getattr(m, "_CONTEXT", None)
    if ctx is None or not hasattr(type(ctx), "get_buffer"):
        return  # this SGLang keeps no such buffers: nothing to move

    def get_buffer(orig):
        def get_buffer(self, name, factory):
            from sglang.srt.layers.moe import kt_park

            def made():
                with kt_park.region():
                    return factory()

            return orig(self, name, made)

        return get_buffer

    wrap(type(ctx), "get_buffer", ["self", "name", "factory"], get_buffer)


after_import("sglang.srt.runtime_context", _buffers)
after_import("sglang.srt.model_executor.model_runner", _runner)
after_import("sglang.srt.managers.scheduler", _scheduler)
after_import("sglang.srt.entrypoints.http_server", _http)
