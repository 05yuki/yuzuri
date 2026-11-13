"""The V4 indexer on a small card: the indexer's [t, H, n] score product is reduced in place, the largest transient of
a prefill there.

The logits budget: SGLang's static budget is a share of what mem_fraction_static leaves of the card. A launch that
gives the static pools the whole card (Yuzuri's lend makes the room a request needs, when it needs it) leaves that
share empty, the budget comes out as one byte and the indexer scores one query row at a time. Where SGLang's share
is empty the budget is SGLang's own ceiling (its share of the card's total, what it uses when no fraction is set);
outside graph capture SGLang then caps it by what the driver has free at that moment, as it always does.
KT_INDEXER_LOGITS_MB (MB) fixes the budget for a launch that wants a cap of its own."""
import os

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/layers/attention/mqa_logits_utils.py", "srt/layers/attention/dsv4/dsv41_sparse.py"]


def _budget(m):
    def make(orig):
        def mqa_logits_static_budget_bytes(*, device_index):
            mb = int(os.environ.get("KT_INDEXER_LOGITS_MB", "0") or 0)
            if mb > 0:
                return mb << 20
            budget = orig(device_index=device_index)
            if budget > 1:
                return budget
            total = m.get_device_module().get_device_properties(device_index).total_memory
            return max(1, int(total * m.MQA_LOGITS_TOTAL_MEM_FRACTION))

        return mqa_logits_static_budget_bytes

    wrap(m, "mqa_logits_static_budget_bytes", ["device_index"], make)


def _scores(m):
    import torch

    def make(orig):
        def scores(self, q, k, weights):
            """q [t, H, d], k [n, d], weights [t, H] -> [t, n], summed over all heads."""
            s = torch.einsum("bhd,nd->bhn", q, k)
            s.relu_().mul_(weights.unsqueeze(-1))
            return s.sum(dim=1).float()

        return scores

    wrap(m.DeepseekV41Indexer, "scores", ["self", "q", "k", "weights"], make)


after_import("sglang.srt.layers.attention.mqa_logits_utils", _budget)
after_import("sglang.srt.layers.attention.dsv4.dsv41_sparse", _scores)
