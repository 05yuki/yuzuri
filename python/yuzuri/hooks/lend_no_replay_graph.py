"""The prefill lend points the layers' weights elsewhere on every prefill step; a decoder replay-tail graph would
keep the addresses of its capture. With the lend on, no such graph is made."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/models/deepseek_v4_replay_tail.py"]


def _install(m):
    def make(orig):
        def maybe_create(cls, model):
            from sglang.srt.layers.moe import kt_lend

            return None if kt_lend.ENABLED else orig(cls, model)

        return maybe_create

    wrap(m.DecoderReplayTail, "maybe_create", ["cls", "model"], make)


after_import("sglang.srt.models.deepseek_v4_replay_tail", _install)
