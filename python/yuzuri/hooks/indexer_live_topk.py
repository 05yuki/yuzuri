"""The torch indexer's decode on the live positions only (kernels/ops/attention/dsv4/live_topk.py, KT_IDX_LIVE_TOPK).

  scoring.decode_scores          the live positions' slots come from one kernel (capacity-sized grid, early exit past
                                 the device-side length) where SGLang gathers and masks the whole width; the result
                                 says so (LiveDecodeScores.live)
  scoring.select_decode          a live result is selected by live_topk_full: same picks, order and slots as SGLang's
                                 topk + sort + gather
  DenseBlocks...consume_decode   a live result is selected among the published blocks by live_topk_among_blocks, ties
                                 in the published block order

Each is a change in the middle of SGLang's function: it is compiled again from its own source with that line changed
(_seam.rewrite), and the launch stops if the line is not there.
"""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import define, rewrite

REPLACES = [
    "srt/layers/attention/dsv4/v41_indexer/dense_blocks.py",
    "srt/layers/attention/dsv4/v41_indexer/scoring.py",
]
_LIVE = '''from sglang.kernels.ops.attention.dsv4.live_topk import (
    live_decode_slots,
    live_topk_among_blocks,
    live_topk_enabled,
    live_topk_full,
)


class LiveDecodeScores(DecodeScores, frozen=True):
    # slots hold [0, lens) only; select with the live kernels
    live: bool = False
'''


def _install_scoring(m):
    define(m, _LIVE, "v41_indexer.scoring live top-k")
    rewrite(m, "decode_scores", [
        ("""    j = torch.arange(lmax, device=pos.device)
    valid = j[None, :] < lens[:, None]
    slots = req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
    slots = slots.masked_fill(~valid, 0)
""", """    live = live_topk_enabled(lmax, pos.device)
    if live:
        slots = live_decode_slots(req_to_token, req, lens, ratio, lmax)
    else:
        j = torch.arange(lmax, device=pos.device)
        valid = j[None, :] < lens[:, None]
        slots = req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
        slots = slots.masked_fill(~valid, 0)
"""),
        ("    return DecodeScores(bs=bs, lmax=lmax, lens=lens, slots=slots, scores=scores)",
         "    return LiveDecodeScores(bs=bs, lmax=lmax, lens=lens, slots=slots, scores=scores, live=live)"),
    ], adds=("live_topk",))
    rewrite(m, "select_decode", [
        ("""    else:
        write_decode(inputs, d, d.scores.topk(k, dim=-1, sorted=False).indices)""",
         """    elif getattr(d, "live", False):
        live_topk_full(d.scores, d.slots, d.lens, k, inputs.out_page_indices[: d.bs])
    else:
        write_decode(inputs, d, d.scores.topk(k, dim=-1, sorted=False).indices)"""),
    ], adds=("live_topk",))


def _install_blocks(m):
    owners = [c for c in vars(m).values() if isinstance(c, type) and c.__module__ == m.__name__
              and "consume_decode" in vars(c)]
    from yuzuri.hooks._seam import SeamError

    if len(owners) != 1:
        raise SeamError(f"yuzuri: {m.__name__} has {len(owners)} classes with consume_decode, the live top-k hook "
                        f"was written for one")
    rewrite(owners[0], "consume_decode", [
        ("""    k = min(inputs.indexer.index_topk, d.lmax)
""", """    k = min(inputs.indexer.index_topk, d.lmax)
    if getattr(d, "live", False):
        from sglang.kernels.ops.attention.dsv4.live_topk import live_topk_among_blocks

        live_topk_among_blocks(d.scores, d.slots, d.lens, published.blocks, k, self.block_size,
                               inputs.out_page_indices[: d.bs])
        return
"""),
    ], adds=("live_topk",))


after_import("sglang.srt.layers.attention.dsv4.v41_indexer.scoring", _install_scoring)
after_import("sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks", _install_blocks)
