#!/usr/bin/env python3
"""The host as a launcher sees it: GPUs with their NUMA node, VRAM and SM count; NUMA nodes with their CPUs and
RAM; CPU ISA flags; the memory settings the launchers depend on (THP, swap, min_free_kbytes); driver / CUDA /
kernel; the model mounts; what is using the GPUs now. JSON on stdout (--json) or a table. No host-specific values
inside: every launcher that still carries a GPU UUID or a CPU count should read this instead.

usage: hostprobe.py [--json] [--gpu-on-node N] [--cpus-of-node N] [--mounts PATH ...]
  --gpu-on-node N   print the UUIDs (comma-separated) of the GPUs on NUMA node N, nothing else
  --cpus-of-node N  print the number of physical cores on NUMA node N (threads to give a CPU expert pool)
Reads /sys, /proc and nvidia-smi only; needs no venv.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys


def _read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def _cpulist(s):
    out = []
    for part in s.split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def numa_nodes():
    nodes = []
    base = "/sys/devices/system/node"
    high = watermarks(_read("/proc/zoneinfo"))
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if not re.fullmatch(r"node\d+", d):
            continue
        n = int(d[4:])
        cpus = _cpulist(_read(f"{base}/{d}/cpulist"))
        # physical cores: one per distinct (core id) among this node's CPUs
        cores = set()
        for c in cpus:
            cores.add(_read(f"/sys/devices/system/cpu/cpu{c}/topology/core_id", str(c)))
        mem = {}
        for line in _read(f"{base}/{d}/meminfo").splitlines():
            m = re.match(r"Node \d+ (\w+):\s+(\d+) kB", line)
            if m:
                mem[m.group(1)] = int(m.group(2)) * 1024
        nodes.append({"node": n, "cpus": cpus, "physical_cores": len(cores), "mem_total": mem.get("MemTotal"),
                      "mem_free": mem.get("MemFree"), "file_pages": mem.get("FilePages"),
                      # what is taken and cannot be reclaimed for a new server: everything that is neither free nor
                      # page cache, and the shared memory counted in the page cache (another server's arenas)
                      "mem_held": max(0, (mem.get("MemTotal") or 0) - (mem.get("MemFree") or 0)
                                      - (mem.get("FilePages") or 0) + (mem.get("Shmem") or 0)),
                      "anon_huge": mem.get("AnonHugePages"),
                      # what a server cannot fill: the zones' high watermarks (below them the kernel reclaims, and a
                      # node of pinned and shared memory has nothing to give back) and what the kernel itself holds
                      "watermark_high": high.get(n, 0),
                      "mem_kernel": max(0, (mem.get("MemTotal") or 0) - (mem.get("MemFree") or 0)
                                        - (mem.get("FilePages") or 0) - (mem.get("AnonPages") or 0))})
    return nodes


def watermarks(zoneinfo, page=None):
    """/proc/zoneinfo -> {node: bytes under its zones' high watermarks}."""
    page = page or os.sysconf("SC_PAGE_SIZE")
    out, node = {}, None
    for line in zoneinfo.splitlines():
        m = re.match(r"Node (\d+), zone", line)
        if m:
            node = int(m.group(1))
            continue
        m = re.match(r"\s+high\s+(\d+)\s*$", line)
        if m and node is not None:
            out[node] = out.get(node, 0) + int(m.group(1)) * page
    return out


def allowed():
    """What this process may use: the CPUs of its affinity, the NUMA nodes its memory may come from (a bind policy
    as numactl -m sets, else the cpuset's nodes), and its cgroup's memory limit (None: no limit)."""
    cpus = sorted(os.sched_getaffinity(0))
    nodes = None
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        mode = ctypes.c_int(0)
        mask = (ctypes.c_ulong * 16)()
        # get_mempolicy (x86-64: 239, aarch64: 236): MPOL_BIND is 2
        nr = {"x86_64": 239, "aarch64": 236}.get(os.uname().machine)
        if nr and libc.syscall(nr, ctypes.byref(mode), mask, ctypes.c_ulong(16 * 8 * ctypes.sizeof(ctypes.c_ulong)),
                               ctypes.c_void_p(0), ctypes.c_ulong(0)) == 0 and mode.value == 2:
            bits = 8 * ctypes.sizeof(ctypes.c_ulong)
            nodes = [i * bits + b for i, w in enumerate(mask) for b in range(bits) if w >> b & 1]
    except Exception:
        nodes = None
    if not nodes:
        for line in _read("/proc/self/status").splitlines():
            if line.startswith("Mems_allowed_list:"):
                nodes = _cpulist(line.split(":", 1)[1].strip())
    limit = None
    for line in _read("/proc/self/cgroup").splitlines():
        if line.startswith("0::"):
            d = "/sys/fs/cgroup" + line[3:].strip()
            while d.startswith("/sys/fs/cgroup"):
                v = _read(os.path.join(d, "memory.max"), "max").strip()
                if v.isdigit():
                    limit = int(v) if limit is None else min(limit, int(v))
                if d == "/sys/fs/cgroup":
                    break
                d = os.path.dirname(d)
    return {"cpus": cpus, "mem_nodes": nodes or [], "memory_max": limit}


def cpu_info():
    text = _read("/proc/cpuinfo")
    flags = set()
    model = ""
    for line in text.splitlines():
        if line.startswith("flags") and not flags:
            flags = set(line.split(":", 1)[1].split())
        elif line.startswith("model name") and not model:
            model = line.split(":", 1)[1].strip()
    isa = {k: (k in flags) for k in ("avx2", "avx512f", "avx512_bf16", "avx512_vnni", "amx_bf16", "amx_int8", "fma")}
    return {"model": model, "isa": isa, "logical_cpus": os.cpu_count()}


def visible(found, spec):
    """The cards a launch may take: all of them, or those CUDA_VISIBLE_DEVICES names, in its order. An entry is a
    card's index as nvidia-smi numbers them (PCI bus order, which the launcher also gives CUDA), or the start of
    its UUID. An entry that names no card stops the launch: a server on another card than the one asked for is
    worse than none."""
    if not spec or not spec.strip():
        return found
    out = []
    for e in (x.strip() for x in spec.split(",") if x.strip()):
        hit = [g for g in found if (e.isdigit() and g["index"] == int(e)) or (not e.isdigit() and g["uuid"].startswith(e))]
        if len(hit) != 1:
            raise SystemExit(f"CUDA_VISIBLE_DEVICES names {e!r}: {len(hit)} of this host's {len(found)} GPUs match")
        if hit[0] not in out:
            out.append(hit[0])
    return out


def gpus():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return [], {}
    q = "index,uuid,pci.bus_id,name,memory.total,memory.used,driver_version,compute_cap"
    out = subprocess.run([smi, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    res = []
    driver = ""
    for line in out.strip().splitlines():
        idx, uuid, bus, name, total, used, driver, cc = [v.strip() for v in line.split(",")]
        bus_l = bus.lower()
        if bus_l.startswith("00000000:"):
            bus_l = bus_l[4:]  # /sys uses the 4-digit domain
        node = _read(f"/sys/bus/pci/devices/{bus_l}/numa_node", "-1")
        res.append({"index": int(idx), "uuid": uuid, "bus": bus, "numa_node": int(node), "name": name,
                    "vram_total": int(total) * 2**20, "vram_used": int(used) * 2**20, "compute_cap": cc})
    # a GPU whose NUMA node the firmware does not report (-1: one-socket desktops do this) is on node 0 when there is
    # one node, else spread over the nodes in GPU order; "numa_node_known" says which
    nodes = sorted(n["node"] for n in numa_nodes()) or [0]
    for g in res:
        g["numa_node_known"] = g["numa_node"] >= 0
        if g["numa_node"] < 0:
            g["numa_node"] = nodes[0] if len(nodes) == 1 else nodes[g["index"] % len(nodes)]
    # a launch asked for some of the cards (two servers on one host, one a card): the others are not this launch's
    res = visible(res, os.environ.get("CUDA_VISIBLE_DEVICES"))
    procs = subprocess.run([smi, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    using = []
    for line in procs.strip().splitlines():
        u, pid, pname, mem = [v.strip() for v in line.split(",")]
        if any(g["uuid"] == u for g in res):  # another server on a card this launch does not take is not in its way
            using.append({"gpu_uuid": u, "pid": int(pid), "name": pname, "used": int(mem) * 2**20})
    leaving = _leaving(smi, res)
    cuda = ""
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    if nvcc:
        v = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+\.\d+)", v)
        cuda = m.group(1) if m else ""
    return res, {"driver": driver, "cuda_toolkit": cuda, "in_use": using, "leaving": leaving}


def _leaving(smi, res):
    """Memory a card holds for no process nvidia-smi lists, and the processes that still have the card's device file
    open without being listed. A server on its way out looks like this: its schedulers are gone from the listing,
    and the driver keeps their memory until the last process with the device file open ends (the launcher, about
    6 s later on a 16 GB card). A running server's other processes (tokenizer, detokenizer) hold the file too, with
    no memory, so the memory says whether anything is leaving. Only this user's processes can be read."""
    listed_mib = {}  # bus -> MiB nvidia-smi lists for the card's processes, every type (G as well as C)
    pids = set()
    out = subprocess.run([smi, "-q", "-d", "PIDS"], capture_output=True, text=True).stdout
    bus = None
    for line in out.splitlines():
        s = line.strip()
        m = re.match(r"^GPU ([0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+)$", s)
        if m:
            bus = m.group(1).lower()
        elif s.startswith("Process ID") and bus:
            pids.add(int(s.split(":")[1]))
        elif s.startswith("Used GPU Memory") and bus and "MiB" in s:
            listed_mib[bus] = listed_mib.get(bus, 0) + int(s.split(":")[1].split()[0])
    minors = {}
    for g in res:
        b = g["bus"].lower()
        info = _read(f"/proc/driver/nvidia/gpus/{b[4:] if b.startswith('00000000:') else b}/information")
        m = re.search(r"Device Minor:\s*(\d+)", info)
        if m:
            minors[f"/dev/nvidia{m.group(1)}"] = g
    holders = {}
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) in pids or int(p) == os.getpid():
            continue
        try:
            fds = os.listdir(f"/proc/{p}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                t = os.readlink(f"/proc/{p}/fd/{fd}")
            except OSError:
                continue
            if t in minors:
                holders.setdefault(minors[t]["uuid"], set()).add(int(p))
    res_by_uuid = {}
    for g in res:
        b = g["bus"].lower()
        res_by_uuid[g["uuid"]] = {"unlisted": max(0, g["vram_used"] - listed_mib.get(b, 0) * 2**20),
                                  "pids": sorted(holders.get(g["uuid"], set()))}
    return [{"gpu_uuid": u, "unlisted": v["unlisted"],
             "pids": [{"pid": p, "name": _read(f"/proc/{p}/comm", "?").strip()} for p in v["pids"]]}
            for u, v in res_by_uuid.items()]


def memory_settings():
    swap = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("SwapTotal"):
            swap = int(line.split()[1]) * 1024
    pick = lambda s: (re.search(r"\[(\w+)\]", s) or [None, s])[1]
    return {"thp_enabled": pick(_read("/sys/kernel/mm/transparent_hugepage/enabled")),
            "thp_shmem": pick(_read("/sys/kernel/mm/transparent_hugepage/shmem_enabled")),
            "thp_defrag": pick(_read("/sys/kernel/mm/transparent_hugepage/defrag")),
            "swap_total": swap, "min_free_kbytes": int(_read("/proc/sys/vm/min_free_kbytes", "0") or 0),
            "overcommit_memory": int(_read("/proc/sys/vm/overcommit_memory", "0") or 0),
            "hugepages_2m": int(_read("/proc/sys/vm/nr_hugepages", "0") or 0)}


def mounts(paths):
    out = []
    for p in paths:
        if not os.path.exists(p):
            out.append({"path": p, "exists": False})
            continue
        st = os.statvfs(p)
        src = ""
        for line in _read("/proc/mounts").splitlines():
            dev, mnt, fstype = line.split()[:3]
            if mnt != "/" and (p == mnt or p.startswith(mnt + "/")) and len(mnt) > len(src):
                src, fs = mnt, fstype
        out.append({"path": p, "exists": True, "free": st.f_bavail * st.f_frsize, "total": st.f_blocks * st.f_frsize,
                    "mount": src or "/"})
    return out


def probe(mount_paths):
    g, gmeta = gpus()
    return {"hostname": os.uname().nodename, "kernel": os.uname().release, "cpu": cpu_info(), "numa": numa_nodes(),
            "allowed": allowed(),
            "gpus": g, "driver": gmeta.get("driver"), "cuda_toolkit": gmeta.get("cuda_toolkit"),
            "gpu_in_use": gmeta.get("in_use", []), "gpu_leaving": gmeta.get("leaving", []), "memory": memory_settings(),
            "load_1m": os.getloadavg()[0], "mounts": mounts(mount_paths)}


def table(h):
    gb = lambda b: f"{(b or 0) / 2**30:.1f} GiB"
    lines = [f"{h['hostname']}  kernel {h['kernel']}  driver {h['driver']}  cuda {h['cuda_toolkit']}  load {h['load_1m']:.1f}",
             f"cpu: {h['cpu']['model']}  logical {h['cpu']['logical_cpus']}  isa " +
             " ".join(k for k, v in h["cpu"]["isa"].items() if v)]
    for n in h["numa"]:
        c = n["cpus"]
        lines.append(f"node {n['node']}: {n['physical_cores']} cores ({len(c)} cpus {c[0]}-{c[-1]}...)  RAM {gb(n['mem_total'])}, "
                     f"free {gb(n['mem_free'])}, page cache {gb(n['file_pages'])}")
    for g in h["gpus"]:
        used = [u for u in h["gpu_in_use"] if u["gpu_uuid"] == g["uuid"]]
        lines.append(f"gpu {g['index']}: {g['name']} cc {g['compute_cap']} bus {g['bus']} node {g['numa_node']} "
                     f"VRAM {gb(g['vram_total'])} used {gb(g['vram_used'])}  {g['uuid']}" +
                     ("  in use by " + ", ".join(f"{u['name']}({u['pid']})" for u in used) if used else ""))
    m = h["memory"]
    lines.append(f"memory: THP {m['thp_enabled']} / shmem {m['thp_shmem']} / defrag {m['thp_defrag']}  swap {gb(m['swap_total'])}  "
                 f"min_free_kbytes {m['min_free_kbytes'] // 1024} MiB  overcommit {m['overcommit_memory']}")
    for mt in h["mounts"]:
        lines.append(f"{mt['path']}: " + (f"free {gb(mt['free'])} of {gb(mt['total'])} ({mt['mount']})" if mt["exists"] else "missing"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--gpu-on-node", type=int)
    ap.add_argument("--cpus-of-node", type=int)
    ap.add_argument("--mounts", nargs="*", default=[os.getcwd()], help="model directories to report free space for")
    a = ap.parse_args()
    h = probe(a.mounts)
    if a.gpu_on_node is not None:
        print(",".join(g["uuid"] for g in h["gpus"] if g["numa_node"] == a.gpu_on_node))
        return
    if a.cpus_of_node is not None:
        n = [x for x in h["numa"] if x["node"] == a.cpus_of_node]
        print(n[0]["physical_cores"] if n else 0)
        return
    print(json.dumps(h, indent=1) if a.json else table(h))


if __name__ == "__main__":
    sys.exit(main())
