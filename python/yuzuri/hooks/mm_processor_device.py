"""SGLANG_MM_PROCESSOR_DEVICE: the device of a multimodal model's fast image processor. serve.py sets cpu: on the GPU
the tokenizer process holds a CUDA context of its own, outside every budget."""
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/multimodal/processors/base_processor.py"]


def _install(m):
    def make(orig):
        def _fast_image_processor_device(self, processor):
            dev = os.environ.get("SGLANG_MM_PROCESSOR_DEVICE")
            return dev if dev else orig(self, processor)

        return _fast_image_processor_device

    wrap(m.BaseMultimodalProcessor, "_fast_image_processor_device", ["self", "processor"], make)


after_import("sglang.srt.multimodal.processors.base_processor", _install)
