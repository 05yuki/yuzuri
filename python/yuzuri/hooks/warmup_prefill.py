"""Two warmups by name (--warmups): prefill_lengths (one fresh prompt of each length in
SGLANG_WARMUP_PREFILL_LENGTHS) and prefill_plan (a setup's first launch measures the plan's chunks and what a
streamed window costs before the server reports ready)."""
from yuzuri._finder import after_import
from yuzuri.hooks._seam import SeamError

REPLACES = ["srt/entrypoints/warmup.py"]


def _install(m):
    for name in ("warmup", "GenerateReqInput", "FAKE_BOOTSTRAP_HOST"):
        if not hasattr(m, name):
            raise SeamError(f"yuzuri: {m.__name__} has no {name}: the prefill warmups have nothing to register with")
    import logging

    import numpy as np

    warmup, GenerateReqInput, FAKE_BOOTSTRAP_HOST = m.warmup, m.GenerateReqInput, m.FAKE_BOOTSTRAP_HOST
    logger = logging.getLogger(m.__name__)

    @warmup("prefill_lengths")
    async def prefill_lengths(disaggregation_mode: str, tokenizer_manager):
        """One fresh prompt of each length in SGLANG_WARMUP_PREFILL_LENGTHS (comma
        separated), before the server reports ready: the first prefill of a
        launch loads every kernel variant its row count selects and allocates the
        staging a streamed prefill keeps, which otherwise lands on the first
        request (seconds on a hybrid CPU/GPU MoE)."""
        import os

        lengths = [
            int(x) for x in os.environ.get("SGLANG_WARMUP_PREFILL_LENGTHS", "").split(",") if x.strip()
        ]
        rng = np.random.default_rng(0)
        for size in lengths:
            logger.info("Warmup prefill of %d tokens", size)
            generate_req_input = GenerateReqInput(
                input_ids=rng.integers(1000, 2**16, size=[size]).tolist(),
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            )
            if disaggregation_mode != "null":
                generate_req_input.bootstrap_room = 0
                generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST
            await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


    @warmup("prefill_plan")
    async def prefill_plan(disaggregation_mode: str, tokenizer_manager):
        """A setup's first launch: the prefill plan (kt_prefill_plan) has no forwards of an earlier launch to plan the
        first window from. Fresh prompts before the server reports ready, each twice the last, from two of the first
        chunk (KT_PREFILL_PLAN_SAFE) to two of the chunk ceiling (KT_PREFILL_LEND_MAX_CHUNK): the plan measures its
        chunks on them, equal chunks along one sequence among them, and leaves them to the next launch in the lend's hint
        file. No longer than the launch has taken so far (the next prompt is skipped when twice the last one's time would
        pass that), and the prompts leave the cache after."""
        import os
        import time

        first = int(os.environ.get("KT_PREFILL_PLAN_SAFE", "0") or 0)
        ceiling = int(os.environ.get("KT_PREFILL_LEND_MAX_CHUNK", "0") or 0)
        if first <= 0 or ceiling <= 0:
            return
        limit = int(getattr(tokenizer_manager, "max_req_input_len", None) or tokenizer_manager.context_len) - 64
        top = min(2 * ceiling, limit)
        lengths, n = [], 2 * first
        while n < top:
            lengths.append(n)
            n *= 2
        if not lengths:
            # (a last prompt of `top` tokens went in three chunks smaller than the ones before it had measured: 28 s of
            # 68 for nothing new. The last doubling's chunks are at least half the ceiling, all the plan's cap asks)
            lengths.append(top)
        try:
            import psutil

            budget = time.time() - psutil.Process().create_time()
        except Exception:
            budget = 600.0
        rng = np.random.default_rng(1)

        def request(size: int, new: int) -> GenerateReqInput:
            r = GenerateReqInput(
                input_ids=rng.integers(1000, 2**16, size=[size]).tolist(),
                sampling_params={"max_new_tokens": new, "temperature": 0.0},
            )
            if disaggregation_mode != "null":
                r.bootstrap_room = 0
                r.bootstrap_host = FAKE_BOOTSTRAP_HOST
            return r

        # a decode step first: a host KV pool's rows come to the card at the first one after an extend (kt_kv_vmm) and
        # the hot lend goes out, the state every served prompt is planned in. Without it the prompts below wrote and
        # read their KV in host memory: 32,768 tokens took 36 s for 5, and the ladder stopped there (Qwen3.5, 11-10)
        async for _ in tokenizer_manager.generate_request(request(64, 4), None):
            pass
        t0, last = time.time(), 0.0

        async def one(size: int) -> None:
            """A prompt alone in the pool, as a served one is: read to its end, then out of the cache. Left in, the
            ladder's prompts stacked up (4,096 to 65,536: 127K tokens held) and the later ones wrote their KV past the
            card's rows, in host memory (a 1M pool, 11-11: 8 forwards)."""
            async for _ in tokenizer_manager.generate_request(request(size, 1), None):
                pass
            try:
                await tokenizer_manager.flush_cache()
            except Exception as e:
                logger.info("Prefill plan: the cache was not flushed (%s)", e)

        # what a window costs (kt_stream_threshold, KT_GPU_STREAM_PREFILL=auto): rounds of two prompts that stream. The
        # resident path's cost comes from the prompts served, not from these random tokens
        from sglang.srt.layers.moe import kt_stream_threshold

        if kt_stream_threshold.AUTO:
            sizes = [n for n in kt_stream_threshold.calibration_lengths() if 0 < n <= limit]
            for _ in range(kt_stream_threshold.REPS):
                for size in sizes:
                    await one(size)
            logger.info("Prefill plan: a window's cost on prompts of %s tokens, %.0f s", sizes, time.time() - t0)
        for size in lengths:
            if time.time() - t0 + 2 * last > budget:
                logger.info("Prefill plan: stopped before %d tokens (the launch took %.0f s)", size, budget)
                break
            logger.info("Prefill plan: a prompt of %d tokens", size)
            t1 = time.time()
            await one(size)
            last = time.time() - t1
        logger.info("Prefill plan: %.0f s", time.time() - t0)


after_import("sglang.srt.entrypoints.warmup", _install)
