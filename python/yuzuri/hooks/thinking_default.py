"""A reasoning parser whose template toggles thinking explicitly (explicit_thinking, explicit_enable_thinking), and a
request without the toggle: SGLANG_DEFAULT_THINKING decides, the fallback the DSV4 encoder uses to open the thinking
block, so the parser and the prompt agree (thinking-default-parser)."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/entrypoints/openai/serving_chat.py"]
EXPLICIT = ("explicit_thinking", "explicit_enable_thinking")


def _install(m):
    from sglang.srt.environ import envs

    def make(orig):
        def _get_reasoning_from_request(self, request):
            got = orig(self, request)
            det = getattr(self, "_reasoning_detector", None)
            mode = getattr(det, "reasoning_default", None)
            if (got or mode not in EXPLICIT or not self.reasoning_parser
                    or self.template_manager.reasoning_config is not None):
                return got
            toggle = mode.replace("explicit_", "")
            kw = request.chat_template_kwargs or {}
            return got if toggle in kw else envs.SGLANG_DEFAULT_THINKING.get() is True

        return _get_reasoning_from_request

    wrap(m.OpenAIServingChat, "_get_reasoning_from_request", ["self", "request"], make)


after_import("sglang.srt.entrypoints.openai.serving_chat", _install)
