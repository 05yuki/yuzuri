"""KT_KV_VMM=1: the KV pool made inside kt_kv_vmm's region, on remappable memory, when torch_memory_saver is off
(sglang-kt-kv-vmm.py)."""
from contextlib import contextmanager

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/utils/torch_memory_saver_adapter.py"]


def _install(m):
    def make(orig):
        @contextmanager
        def region(self, tag, enable_cpu_backup=False):
            if tag == "kv_cache":
                from sglang.srt.layers.moe import kt_kv_vmm

                if kt_kv_vmm.ENABLED:
                    with kt_kv_vmm.region():
                        yield
                    return
            with orig(self, tag, enable_cpu_backup=enable_cpu_backup):
                yield

        return region

    wrap(m._TorchMemorySaverAdapterNoop, "region", ["self", "tag", "enable_cpu_backup"], make)


after_import("sglang.srt.utils.torch_memory_saver_adapter", _install)
