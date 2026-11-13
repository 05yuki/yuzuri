#!/usr/bin/env python3
"""A hardware report to paste into an issue: what Yuzuri reads to choose its kernels and sizes, without anything that
names the machine or its owner (no hostname, GPU UUID, user name, home path or other processes' names).

usage: python tools/diag.py [--json] [--bench] [--model-dir PATH] [--recipe PATH]
  --json        the whole report as JSON instead of the text summary
  --bench       also measure, briefly (about 30 s, GPUs and CPUs otherwise idle): pinned host <-> GPU copy rates
                and each NUMA node's memory copy rate. Needs torch (the recipe's venv, or any).
  --model-dir   a model directory: its filesystem, free space and whether its disk is rotational (the path itself
                is not reported)
  --recipe      the recipe checkout: the versions of its venv (torch, SGLang, kt-kernel and its CPU variant)

Reads /sys, /proc and nvidia-smi; the versions and --bench import torch when they can. Run it on an idle machine:
a running server changes the free memory and the bench.
"""
import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hostprobe  # noqa: E402

ISA = ("sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512_bf16",
       "avx512_fp16", "avx_vnni", "amx_tile", "amx_bf16", "amx_int8", "amx_fp16")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def cpu():
    text = hostprobe._read("/proc/cpuinfo")
    flags, model, sockets = set(), "", set()
    for line in text.splitlines():
        if line.startswith("flags") and not flags:
            flags = set(line.split(":", 1)[1].split())
        elif line.startswith("model name") and not model:
            model = line.split(":", 1)[1].strip()
        elif line.startswith("physical id"):
            sockets.add(line.split(":", 1)[1].strip())
    # last-level caches: one per distinct shared CPU list (a CCD on AMD, a socket's L3 on most Intel parts)
    l3 = {}
    base = "/sys/devices/system/cpu"
    for d in os.listdir(base) if os.path.isdir(base) else []:
        if not re.fullmatch(r"cpu\d+", d):
            continue
        for idx in ("index3", "index2"):
            p = f"{base}/{d}/cache/{idx}"
            if hostprobe._read(f"{p}/level") == idx[-1]:
                l3.setdefault((idx, hostprobe._read(f"{p}/shared_cpu_list")), hostprobe._read(f"{p}/size"))
                break
    llc = sorted({(k[0], v) for k, v in l3.items()})
    freq = hostprobe._read(f"{base}/cpu0/cpufreq/cpuinfo_max_freq")
    return {"model": model, "sockets": len(sockets) or 1, "logical_cpus": os.cpu_count(),
            "isa": [k for k in ISA if k in flags],
            "last_level_caches": len(l3), "last_level_cache_size": llc[0][1] if llc else "",
            "max_mhz": int(freq) // 1000 if freq.isdigit() else None}


def numa():
    return [{"node": n["node"], "physical_cores": n["physical_cores"], "logical_cpus": len(n["cpus"]),
             "ram_gib": round((n["mem_total"] or 0) / 2**30, 1), "free_gib": round((n["mem_free"] or 0) / 2**30, 1)}
            for n in hostprobe.numa_nodes()]


def gpus():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return [], {}
    q = ("index,pci.bus_id,name,memory.total,memory.used,compute_cap,pcie.link.gen.max,pcie.link.gen.current,"
         "pcie.link.width.max,pcie.link.width.current,power.limit,clocks.max.sm,driver_version")
    out = _run([smi, f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    res, driver = [], ""
    for line in out.strip().splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) < 13:
            continue
        bus = v[1].lower()
        bus = bus[4:] if bus.startswith("00000000:") else bus
        driver = v[12]
        num = lambda x: int(float(x)) if re.fullmatch(r"[\d.]+", x) else None
        res.append({"index": int(v[0]), "name": v[2], "compute_cap": v[5],
                    "vram_gib": round(int(v[3]) / 1024, 1), "vram_used_gib": round(int(v[4]) / 1024, 1),
                    "numa_node": int(hostprobe._read(f"/sys/bus/pci/devices/{bus}/numa_node", "-1")),  # -1: not reported
                    "pcie_gen_max": num(v[6]), "pcie_gen_now": num(v[7]),
                    "pcie_width_max": num(v[8]), "pcie_width_now": num(v[9]),
                    "power_limit_w": num(v[10]), "sm_clock_max_mhz": num(v[11])})
    # how each pair of cards reaches the other (P2P read capability, the topology letters)
    p2p = {}
    m = _run([smi, "topo", "-p2p", "r"])
    rows = [ln.split() for ln in m.splitlines() if re.match(r"\s*GPU\d+", ln)]
    for r in rows:
        i = int(r[0][3:])
        for j, cell in enumerate(r[1:1 + len(rows)]):
            if i != j:
                p2p[f"{i}-{j}"] = cell
    topo = {}
    t = _run([smi, "topo", "-m"])
    trows = [ln.split() for ln in t.splitlines() if re.match(r"GPU\d+\s", ln)]
    for r in trows:
        i = int(r[0][3:])
        for j, cell in enumerate(r[1:1 + len(trows)]):
            if i != j:
                topo[f"{i}-{j}"] = cell
    busy = _run([smi, "--query-compute-apps=pid", "--format=csv,noheader"]).split()
    return res, {"driver": driver, "p2p_read": p2p, "link": topo, "processes_on_gpus": len(busy)}


def software(recipe):
    osr = {}
    for line in hostprobe._read("/etc/os-release").splitlines():
        k, _, v = line.partition("=")
        osr[k] = v.strip('"')
    sw = {"os": osr.get("PRETTY_NAME", ""), "kernel": platform.release(),
          "glibc": "-".join(platform.libc_ver()) or "", "python": platform.python_version()}
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    if nvcc:
        m = re.search(r"release (\d+\.\d+)", _run([nvcc, "--version"]))
        sw["cuda_toolkit"] = m.group(1) if m else ""
    py = os.path.join(recipe, ".venv", "bin", "python") if recipe else sys.executable
    code = ("import json,importlib.metadata as md\n"
            "out={}\n"
            "for p in ('torch','sglang','sglang-kernel','sgl-kernel','kt-kernel','flashinfer-python','triton','tilelang'):\n"
            "    try: out[p]=md.version(p)\n"
            "    except Exception: pass\n"
            "try:\n"
            "    import torch; out['torch_cuda']=torch.version.cuda\n"
            "except Exception: pass\n"
            "try:\n"
            "    import kt_kernel; out['kt_kernel_cpu_variant']=getattr(kt_kernel,'__cpu_variant__',None)\n"
            "except Exception as e: out['kt_kernel_import']=type(e).__name__\n"
            "print(json.dumps(out))\n")
    if os.path.exists(py):
        env = dict(os.environ)
        if recipe:
            # kt-kernel's hwloc and the recipe's small libraries, as serve.py gives them
            lib = os.path.join(recipe, "prebuilt", "lib")
            env["LD_LIBRARY_PATH"] = lib + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        try:
            sw["packages"] = json.loads(subprocess.run([py, "-I", "-c", code], capture_output=True, text=True,
                                                       timeout=120, env=env).stdout.strip().splitlines()[-1])
        except (ValueError, IndexError, OSError, subprocess.TimeoutExpired):
            sw["packages"] = {}
    return sw


def model_disk(path):
    if not path or not os.path.exists(path):
        return None
    st = os.statvfs(path)
    best, dev, fs = "", "", ""
    for line in hostprobe._read("/proc/mounts").splitlines():
        d, mnt, fstype = line.split()[:3]
        if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
            best, dev, fs = mnt, d, fstype
    rot = None
    name = os.path.basename(os.path.realpath(dev)) if dev.startswith("/dev/") else ""
    for cand in (name, re.sub(r"p?\d+$", "", name)):
        r = hostprobe._read(f"/sys/block/{cand}/queue/rotational")
        if r in ("0", "1"):
            rot = r == "1"
            break
    return {"filesystem": fs, "rotational": rot, "free_gib": round(st.f_bavail * st.f_frsize / 2**30, 1),
            "total_gib": round(st.f_blocks * st.f_frsize / 2**30, 1)}


def memory():
    m = hostprobe.memory_settings()
    return {"thp": m["thp_enabled"], "thp_shmem": m["thp_shmem"], "swap_gib": round(m["swap_total"] / 2**30, 1),
            "min_free_mib": m["min_free_kbytes"] // 1024, "hugepages_2m": m["hugepages_2m"],
            "ptrace_scope": hostprobe._read("/proc/sys/kernel/yama/ptrace_scope", "")}


def bench(recipe, nodes):
    """Pinned host <-> GPU copy rates per GPU, and each NUMA node's memory copy rate (one process per node, its
    cores, a 2 GiB copy): rough, for comparing machines, not a benchmark."""
    py = os.path.join(recipe, ".venv", "bin", "python") if recipe else sys.executable
    gpu_code = ("import json,torch,time\n"
                "out=[]\n"
                "for i in range(torch.cuda.device_count()):\n"
                "    d=torch.device('cuda',i); n=1<<30\n"
                "    h=torch.empty(n,dtype=torch.uint8,pin_memory=True); g=torch.empty(n,dtype=torch.uint8,device=d)\n"
                "    for _ in range(2): g.copy_(h); h.copy_(g)\n"
                "    torch.cuda.synchronize(d)\n"
                "    t=time.perf_counter(); [g.copy_(h,non_blocking=True) for _ in range(4)]; torch.cuda.synchronize(d); h2d=4*n/(time.perf_counter()-t)\n"
                "    [g.copy_(h,non_blocking=True) for _ in range(16)]\n"
                "    import subprocess as sp\n"
                "    link=sp.run(['nvidia-smi','-i',str(i),'--query-gpu=pcie.link.gen.current,pcie.link.width.current','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()\n"
                "    torch.cuda.synchronize(d)\n"
                "    t=time.perf_counter(); [h.copy_(g,non_blocking=True) for _ in range(4)]; torch.cuda.synchronize(d); d2h=4*n/(time.perf_counter()-t)\n"
                "    out.append({'index':i,'h2d_gbs':round(h2d/1e9,1),'d2h_gbs':round(d2h/1e9,1),'link_under_load':link})\n"
                "print(json.dumps(out))\n")
    res = {}
    try:
        res["gpu_copy"] = json.loads(_run([py, "-I", "-c", gpu_code]).strip().splitlines()[-1])
    except (ValueError, IndexError):
        res["gpu_copy"] = "torch with CUDA not found"
    cpu_code = ("import json,torch,time,sys\n"
                "torch.set_num_threads(int(sys.argv[1]))\n"
                "n=1<<31; a=torch.ones(n//4,dtype=torch.float32); b=torch.empty_like(a)\n"
                "b.copy_(a)\n"
                "t=time.perf_counter()\n"
                "for _ in range(5): b.copy_(a)\n"
                "print(json.dumps(round(5*2*n/(time.perf_counter()-t)/1e9,1)))\n")
    numactl = shutil.which("numactl")
    per = []
    for n in nodes:
        cmd = ([numactl, f"--cpunodebind={n['node']}", f"--membind={n['node']}"] if numactl else []) + \
              [py, "-I", "-c", cpu_code, str(max(1, n["physical_cores"]))]
        try:
            per.append({"node": n["node"], "copy_gbs": json.loads(_run(cmd).strip().splitlines()[-1]),
                        "bound": bool(numactl)})
        except (ValueError, IndexError):
            per.append({"node": n["node"], "copy_gbs": None})
    res["node_copy"] = per
    return res


def text(r):
    c = r["cpu"]
    lines = [f"Yuzuri hardware report {r['date']}",
             f"CPU: {c['model']}, {c['sockets']} socket(s), {c['logical_cpus']} logical CPUs, "
             f"{c['last_level_caches']} last-level caches of {c['last_level_cache_size']}, max {c['max_mhz']} MHz",
             f"ISA: {' '.join(c['isa'])}"]
    for n in r["numa"]:
        lines.append(f"NUMA node {n['node']}: {n['physical_cores']} cores / {n['logical_cpus']} CPUs, "
                     f"RAM {n['ram_gib']} GiB ({n['free_gib']} free)")
    for g in r["gpus"]:
        lines.append(f"GPU {g['index']}: {g['name']}, cc {g['compute_cap']}, {g['vram_gib']} GiB "
                     f"({g['vram_used_gib']} used), NUMA node {g['numa_node'] if g['numa_node'] >= 0 else 'not reported'}, "
                     f"PCIe gen {g['pcie_gen_now']}/"
                     f"{g['pcie_gen_max']} x{g['pcie_width_now']}/{g['pcie_width_max']}, {g['power_limit_w']} W")
    gm = r["gpu_meta"]
    if gm.get("link"):
        lines.append("GPU links: " + ", ".join(f"{k} {v}" for k, v in sorted(gm["link"].items())))
    if gm.get("p2p_read"):
        lines.append("GPU P2P read: " + ", ".join(f"{k} {v}" for k, v in sorted(gm["p2p_read"].items())))
    lines.append(f"driver {gm['driver']}, processes on the GPUs: {gm.get('processes_on_gpus', 0)}"
                 if gm.get("driver") else "no NVIDIA GPU found (nvidia-smi)")
    s = r["software"]
    lines.append(f"OS: {s['os']}, kernel {s['kernel']}, {s['glibc']}, Python {s['python']}, "
                 f"CUDA toolkit {s.get('cuda_toolkit', 'none')}")
    if s.get("packages"):
        lines.append("packages: " + ", ".join(f"{k} {v}" for k, v in s["packages"].items()))
    m = r["memory"]
    lines.append(f"memory: THP {m['thp']} (shmem {m['thp_shmem']}), swap {m['swap_gib']} GiB, "
                 f"min_free {m['min_free_mib']} MiB, 2M hugepages {m['hugepages_2m']}")
    if r.get("model_disk"):
        d = r["model_disk"]
        lines.append(f"model disk: {d['filesystem']}, "
                     f"{'rotational' if d['rotational'] else 'solid state' if d['rotational'] is False else 'unknown'}, "
                     f"{d['free_gib']} of {d['total_gib']} GiB free")
    if r.get("bench"):
        b = r["bench"]
        if isinstance(b.get("gpu_copy"), list):
            for g in b["gpu_copy"]:
                lines.append(f"bench GPU {g['index']}: host->GPU {g['h2d_gbs']} GB/s, GPU->host {g['d2h_gbs']} GB/s, "
                             f"PCIe gen, width while copying: {g.get('link_under_load', '')}")
        for n in b.get("node_copy", []):
            lines.append(f"bench node {n['node']}: memory copy {n['copy_gbs']} GB/s"
                         + ("" if n.get("bound", True) else " (numactl missing: not bound to the node)"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--model-dir")
    ap.add_argument("--recipe")
    a = ap.parse_args()
    g, gm = gpus()
    r = {"date": datetime.date.today().isoformat(), "cpu": cpu(), "numa": numa(), "gpus": g, "gpu_meta": gm,
         "software": software(a.recipe), "memory": memory(), "model_disk": model_disk(a.model_dir)}
    if a.bench:
        if gm.get("processes_on_gpus"):
            print("note: processes are using the GPUs; the bench numbers will be low", file=sys.stderr)
        r["bench"] = bench(a.recipe, r["numa"])
    out = json.dumps(r, indent=1) if a.json else text(r)
    # nothing that names the machine or its owner: the home directory and the user name never leave
    home, user = os.path.expanduser("~"), os.environ.get("USER", "")
    out = out.replace(home, "~")
    if user and len(user) > 2:
        out = re.sub(rf"\b{re.escape(user)}\b", "<user>", out)
    print(out)


if __name__ == "__main__":
    sys.exit(main())
