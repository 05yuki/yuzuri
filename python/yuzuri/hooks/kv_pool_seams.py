"""The KV pool configurator's part in the lend, the host KV and the state-slot count, on four of its methods:

  _resolve_memory_pool_config  the sliding-window pool is budgeted at the lend's own chunk (kt_kv_prefill_lend), and
                               the config the budget gave (config_from_budget, inside this call only) is made to
                               cover the context: a pool that holds fewer tokens than the context is made at the
                               context, its tensors in host memory (kt_kv_vmm.HOST), and the card takes their first
                               rows once the server is up
  _derive_pool_sizes           the sliding-window pool holds the largest planned chunk
  _init_pools                  the lend learns the allocator and the pool it lends from
  _handle_max_mamba_cache      KT_MAMBA_REQUESTS: so many requests' state slots, as SGLang counts a request's
"""
import os
import threading

from yuzuri._finder import after_import
from yuzuri.hooks._seam import wrap

REPLACES = ["srt/mem_cache/kv_cache_configurator.py"]
_IN = threading.local()


def _cover_context(m, self, config):
    """A pool that holds fewer tokens than the context is made at the context. Whatever the configurator: it sizes
    the pool for the token count as it would for any other."""
    from sglang.srt.layers.moe import kt_kv_vmm

    ctx = int(self.model_config.context_len)
    page = m.get_schedule().page_size
    tokens = -(-ctx // page) * page
    if os.environ.get("KT_HOST_KV", "1") == "0" or not kt_kv_vmm.ENABLED:
        return config
    # a pool that splits itself between the card and host memory (compressed sub-pools) is left to it; any other is
    # made in host memory even when the budget covers the context, so the buffers made after it are measured before
    # the card takes its rows
    if config.max_total_num_tokens >= ctx and (int(getattr(config, "c4_max_total_num_tokens", 0) or 0)
                                               or int(getattr(config, "c128_max_total_num_tokens", 0) or 0)):
        return config
    from sglang.srt.model_executor.pool_configurator import create_memory_pool_configurator

    before = config.max_total_num_tokens
    budget_config = config
    config = create_memory_pool_configurator(self).calculate_pool_sizes_from_max_tokens(tokens, page)
    # the other pools keep the size the card's budget gave them (a window pool needs its window and a chunk, not the
    # context)
    for k, v in list(vars(config).items()):
        if (k.endswith("_max_total_num_tokens") and k != "full_max_total_num_tokens" and isinstance(v, int)
                and isinstance(getattr(budget_config, k, None), int) and 0 < getattr(budget_config, k) < v):
            setattr(config, k, getattr(budget_config, k))
    full = int(getattr(config, "full_max_total_num_tokens", None) or config.max_total_num_tokens)
    # a pool tensor's leading dimension: tokens (with the padding page) or pages (with the reserved one)
    rows = {full, full + page, full // page, full // page + 1}
    kt_kv_vmm.HOST = {"rows": rows, "rows_of_pool": full + page, "full": full, "page": page,
                      "vocab": int(getattr(self.model_config, "vocab_size", 0) or 0)}
    if os.environ.get("KT_HOST_KV_DEBUG") == "1":
        m.logger.info("host KV sizes: " + ", ".join(
            f"{k} {getattr(budget_config, k, None)} -> {v}" for k, v in vars(config).items()
            if k.endswith("max_total_num_tokens")))
    m.logger.info(f"host KV: the pool covers the context, {before} -> {full} tokens; its tensors start in host "
                  f"memory and the card takes their first rows once the server is up")
    return config


def _install(m):
    K = m.KVCacheConfigurator

    def make_resolve(orig):
        def _resolve_memory_pool_config(self, pre_model_load_memory):
            from sglang.srt.layers.moe import kt_kv_prefill_lend

            kt_kv_prefill_lend.swa_auto(self.is_hybrid_swa)
            _IN.depth = getattr(_IN, "depth", 0) + 1
            try:
                with kt_kv_prefill_lend.swa_sizing():
                    return orig(self, pre_model_load_memory)
            finally:
                _IN.depth -= 1

        return _resolve_memory_pool_config

    def make_budget(orig):
        def config_from_budget(self, budget_bytes, *args, **kwargs):
            config = orig(self, budget_bytes, *args, **kwargs)
            if getattr(_IN, "depth", 0) > 0:
                config = _cover_context(m, self, config)
            return config

        return config_from_budget

    def make_sizes(orig):
        def _derive_pool_sizes(self, *args, **kwargs):
            sizes = orig(self, *args, **kwargs)
            if sizes.swa_max_total_num_tokens is not None and not self.is_draft_worker:
                import msgspec

                from sglang.srt.layers.moe import kt_kv_prefill_lend

                swa, c4 = kt_kv_prefill_lend.swa_size(
                    sizes.swa_max_total_num_tokens, self.pool_page_size, sizes.c4_state_pool_size)
                sizes = msgspec.structs.replace(sizes, swa_max_total_num_tokens=swa, c4_state_pool_size=c4)
            return sizes

        return _derive_pool_sizes

    def make_pools(orig):
        def _init_pools(self, *args, **kwargs):
            pools = orig(self, *args, **kwargs)
            if self.is_hybrid_swa and not self.is_draft_worker:
                from sglang.srt.layers.moe import kt_kv_prefill_lend

                kt_kv_prefill_lend.swa_init(pools.token_to_kv_pool_allocator, pools.token_to_kv_pool)
            return pools

        return _init_pools

    def make_mamba(orig):
        def _handle_max_mamba_cache(self, total_rest_memory):
            n = int(os.environ.get("KT_MAMBA_REQUESTS", "0") or 0)
            if n > 0 and m.get_schedule().max_mamba_cache_size is None:
                m.get_context().override(
                    "mamba_pool.kt_requests",
                    max_mamba_cache_size=n * self._calculate_mamba_ratio() * self.attn_dp_size)
                m.logger.info(f"linear-attention state pool for {n} requests: "
                              f"{m.get_schedule().max_mamba_cache_size} slots "
                              f"({self._calculate_mamba_ratio()} a request)")
            return orig(self, total_rest_memory)

        return _handle_max_mamba_cache

    wrap(K, "_resolve_memory_pool_config", ["self", "pre_model_load_memory"], make_resolve,
         adds=("kt_kv_prefill_lend", "_host_kv_cover_context"), needs=("self.config_from_budget(",))
    wrap(K, "config_from_budget", ["self", "budget_bytes", "..."], make_budget, adds=("kt_kv_vmm",))
    wrap(K, "_derive_pool_sizes", ["self", "config"], make_sizes, adds=("kt_kv_prefill_lend",),
         needs=("swa_max_total_num_tokens", "c4_state_pool_size"))
    wrap(K, "_init_pools", ["self", "sizes", "..."], make_pools, adds=("kt_kv_prefill_lend",))
    wrap(K, "_handle_max_mamba_cache", ["self", "total_rest_memory"], make_mamba, adds=("KT_MAMBA_REQUESTS",),
         needs=("max_mamba_cache_size is not None",))


after_import("sglang.srt.mem_cache.kv_cache_configurator", _install)
