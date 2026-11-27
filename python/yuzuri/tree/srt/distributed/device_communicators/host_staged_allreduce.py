"""All-reduce for two GPUs without peer access, staged through pinned host
memory with the copy engines, driven entirely from the GPU streams.

NCCL's SHM transport on this host moves a 20 MB all-reduce at 1.9 GB/s (its
kernels read host memory over PCIe one line at a time; NCCL_SHM_USE_CUDA_MEMCPY
fixes that but doubles the latency of the 10 KB decode all-reduces). Here
each rank DMAs its input to a shared pinned buffer in pieces and publishes a
counter with cuStreamWriteValue32; the peer's stream waits on that counter
with cuStreamWaitValue32, DMAs the piece back into a double-buffered device
staging slot, and the compute stream adds it while the H2D copy engine moves
immediately to the next region. Each rank faults its own send regions on its
local NUMA node on 2 MiB transparent huge pages before cudaHostRegister so D2H
writes stay local and IOMMU/TLB overhead is minimized.

Used only for messages of at least SGLANG_HOST_STAGED_ALLREDUCE_MIN_BYTES
(default 1 MiB) outside CUDA graph capture; decode keeps NCCL. Enable with
SGLANG_HOST_STAGED_ALLREDUCE=1.
"""
import ctypes
import logging
import mmap
import os
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

_FLAG_BYTES = 4096  # one page of uint32 flags at the head of the mapping
_HUGE_ALIGN = 2 << 20  # 2 MiB alignment so data regions sit on THP boundaries
_NBUF = 2  # double buffer per rank (2 x 16 MiB = 32 MiB device staging, zero extra VRAM)
_REGION = int(os.environ.get("SGLANG_HOST_STAGED_ALLREDUCE_REGION", str(16 << 20)))
_PIECE = int(os.environ.get("SGLANG_HOST_STAGED_ALLREDUCE_PIECE", str(8 << 20)))
_MADV_HUGEPAGE = 14
_MADV_COLLAPSE = 25

_cuda = None
_libc = None


def _libcuda():
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
        _cuda.cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
        _cuda.cuStreamWriteValue32_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        _cuda.cuStreamWaitValue32_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
    return _cuda


def _madvise(addr: int, length: int, advice: int) -> int:
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    return int(_libc.madvise(addr, length, advice))


class HostStagedAllReduce:
    def __init__(self, group, rank_in_group: int, world_size: int, device: torch.device,
                 region_bytes: int = _REGION):
        assert world_size == 2, "host-staged all-reduce is written for two ranks"
        self.rank = rank_in_group
        self.peer = 1 - rank_in_group
        self.device = device
        # Round region_bytes up to a 2 MiB multiple so every region is hugepage-aligned.
        self.region_bytes = ((region_bytes + _HUGE_ALIGN - 1) // _HUGE_ALIGN) * _HUGE_ALIGN
        self.min_bytes = int(os.environ.get("SGLANG_HOST_STAGED_ALLREDUCE_MIN_BYTES", str(1 << 20)))
        self._flag_off = _HUGE_ALIGN
        # Both ranks must agree on the file; rank 0 picks the name and the
        # gloo group carries it (init time only, before any forward).
        name = [f"/dev/shm/sgl_host_ar_{os.getpid()}_{time.time_ns()}" if rank_in_group == 0 else None]
        src = torch.distributed.get_process_group_ranks(group)[0]
        torch.distributed.broadcast_object_list(name, src=src, group=group)
        self.path = name[0]
        self.total = self._flag_off + world_size * _NBUF * self.region_bytes
        if rank_in_group == 0:
            with open(self.path, "wb") as f:
                f.truncate(self.total)
        torch.distributed.barrier(group=group)
        fd = os.open(self.path, os.O_RDWR)
        self.mm = mmap.mmap(fd, self.total)
        os.close(fd)
        buf = np.frombuffer(self.mm, dtype=np.uint8)
        addr = buf.ctypes.data
        # Request 2 MiB transparent huge pages before first-touch so shmem_enabled=advise allocates THP directly.
        _madvise(addr, self.total, _MADV_HUGEPAGE)
        # First-touch each rank's own D2H send regions on its local NUMA node before pinning:
        # local D2H + remote H2D balances both sockets' memory channels (~40% faster on dual-socket EPYC).
        if rank_in_group == 0:
            buf[:self._flag_off:4096] = 0
        r0 = self._flag_off + rank_in_group * _NBUF * self.region_bytes
        r1 = r0 + _NBUF * self.region_bytes
        buf[r0:r1:4096] = 0
        torch.distributed.barrier(group=group)
        if rank_in_group == 0:
            _madvise(addr, self.total, _MADV_COLLAPSE)
        torch.distributed.barrier(group=group)

        self.flags = np.frombuffer(self.mm, dtype=np.uint32, count=_FLAG_BYTES // 4)
        with torch.cuda.device(device):
            err = torch.cuda.cudart().cudaHostRegister(addr, self.total, 3)  # Portable | Mapped
            if err != 0:
                raise RuntimeError(f"cudaHostRegister failed: {err}")
            dptr = ctypes.c_uint64()
            rc = _libcuda().cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), addr, 0)
            if rc != 0:
                raise RuntimeError(f"cuMemHostGetDevicePointer failed: {rc}")
        self.dflags = dptr.value  # device address of the flag page
        self.host = torch.from_numpy(buf)
        self.regions = [[self._region(r, i) for i in range(_NBUF)] for r in range(world_size)]
        self.tmps = [torch.empty(self.region_bytes, dtype=torch.uint8, device=device) for _ in range(_NBUF)]
        self.seq = 0
        self.published = 0  # pieces this rank has published, ever
        self.d2h = torch.cuda.Stream(device=device)
        self.h2d = torch.cuda.Stream(device=device)
        self._d2h_events = [torch.cuda.Event() for _ in range(_NBUF)]
        self._h2d_events = [torch.cuda.Event() for _ in range(_NBUF)]
        self._add_events = [torch.cuda.Event() for _ in range(_NBUF)]
        if rank_in_group == 0:
            self.flags[:] = 0
        torch.distributed.barrier(group=group)
        if rank_in_group == 1:
            os.unlink(self.path)  # the mappings keep it alive
        logger.info("host-staged all-reduce ready: %s, %d MB/region (x%d tmp), %d MB pieces, min %d KB (THP+NUMA)",
                    self.path, self.region_bytes >> 20, _NBUF, _PIECE >> 20, self.min_bytes >> 10)

    def _region(self, r, i):
        off = self._flag_off + (r * _NBUF + i) * self.region_bytes
        return self.host[off:off + self.region_bytes]

    # flags (uint32, monotonic): [r] = pieces rank r has landed in host
    # memory, [2 + r] = all-reduces rank r has finished reading back.
    def _write(self, stream, idx, value):
        rc = _libcuda().cuStreamWriteValue32_v2(stream.cuda_stream, self.dflags + 4 * idx, value, 0)
        if rc != 0:
            raise RuntimeError(f"cuStreamWriteValue32 failed: {rc}")

    def _wait(self, stream, idx, value):
        rc = _libcuda().cuStreamWaitValue32_v2(stream.cuda_stream, self.dflags + 4 * idx, value, 0)  # GEQ
        if rc != 0:
            raise RuntimeError(f"cuStreamWaitValue32 failed: {rc}")

    def should_use(self, t: torch.Tensor) -> bool:
        return (t.is_cuda and t.is_contiguous()
                and t.numel() * t.element_size() >= self.min_bytes
                and not torch.cuda.is_current_stream_capturing())

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        flat = t.view(-1).view(torch.uint8)
        nbytes = flat.numel()
        cur = torch.cuda.current_stream(self.device)
        self.d2h.wait_stream(cur)
        self.h2d.wait_stream(cur)
        r, p = self.rank, self.peer
        piece = _PIECE
        if nbytes < 4 * piece:
            piece = max(1 << 20, (nbytes // 4) & ~4095)
        for off in range(0, nbytes, self.region_bytes):
            n = min(self.region_bytes, nbytes - off)
            self.seq += 1
            seq = self.seq
            slot = seq % _NBUF
            mine = self.regions[r][slot]
            theirs = self.regions[p][slot]
            tmp = self.tmps[slot]
            src = flat[off:off + n]
            pieces = [(a, min(piece, n - a)) for a in range(0, n, piece)]
            # the peer must have read this host slot back from two regions ago
            if seq > _NBUF:
                self._wait(self.d2h, 2 + p, seq - _NBUF)
            with torch.cuda.stream(self.d2h):
                for a, m in pieces:
                    mine[a:a + m].copy_(src[a:a + m], non_blocking=True)
                    self.published += 1
                    self._write(self.d2h, r, self.published)
                self._d2h_events[slot].record(self.d2h)
            base = self.published - len(pieces)
            with torch.cuda.stream(self.h2d):
                if seq > _NBUF:
                    self.h2d.wait_event(self._add_events[slot])
                for i, (a, m) in enumerate(pieces):
                    self._wait(self.h2d, p, base + i + 1)
                    tmp[a:a + m].copy_(theirs[a:a + m], non_blocking=True)
                self._write(self.h2d, 2 + r, seq)
                self._h2d_events[slot].record(self.h2d)
            # Our own pieces must have left src before the add overwrites it;
            # running add_ on cur lets h2d start the next region immediately.
            cur.wait_event(self._d2h_events[slot])
            cur.wait_event(self._h2d_events[slot])
            src.view(t.dtype).add_(tmp[:n].view(t.dtype))
            self._add_events[slot].record(cur)
        return t
