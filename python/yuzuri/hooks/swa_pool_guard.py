"""A sliding-window pool smaller than one prefill chunk plus a page seats no request: the scheduler re-queues the
one it cannot admit and spins on one core with the card idle and no prefill logged. The launch stops with the sizes
instead (the hybrid guard SGLang has does not fire where sliding_window_size is None)."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/model_executor/pool_configurator.py"]


def _install(m):
    def make(orig):
        def _compute_dsv4_sizes(self, full_token, page_size):

            got = orig(self, full_token, page_size)
            chunk = m.get_schedule().chunked_prefill_size
            swa = got.swa_max_total_num_tokens
            if chunk and chunk > 0 and swa < chunk + page_size:
                raise ValueError(
                    f"SWA pool ({swa} tokens) is smaller than one prefill chunk ({chunk}) plus a page ({page_size}), "
                    f"so no request can be admitted. The pool is full_tokens ({got.full_max_total_num_tokens}) * "
                    f"swa_full_tokens_ratio ({getattr(self, 'swa_ratio', '?')}); raise --swa-full-tokens-ratio or "
                    f"--max-total-tokens, or lower --chunked-prefill-size.")
            return got

        return _compute_dsv4_sizes

    wrap(m.DSV4PoolConfigurator, "_compute_dsv4_sizes", ["self", "full_token", "page_size"], make)


after_import("sglang.srt.model_executor.pool_configurator", _install)
