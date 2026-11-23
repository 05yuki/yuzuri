// A CUDA allocator for torch.cuda.CUDAPluggableAllocator whose allocations are
// one reserved virtual range each, backed by physical chunks of KV_VMM_CHUNK_MB
// (default 2 MiB, rounded to the device's granularity) that can be unmapped and
// mapped again while the virtual range, and so every tensor and every captured
// CUDA graph that points into it, stays where it is. For the KV pool: the pages
// no request holds give their physical memory to a streamed prefill and get it
// back, zeroed, before decode (kt_kv_vmm.py).
//
//   kv_vmm_release(ptr, off, len)   unmap and release every chunk wholly inside
//                                   [ptr + off, ptr + off + len); returns the
//                                   bytes given back (the caller syncs first)
//   kv_vmm_restore(ptr, off, len)   map new, zeroed chunks wherever that range
//                                   has none; returns the bytes mapped, or -1
//                                   when the driver has no memory for them
//   kv_vmm_released_bytes(device)   bytes currently unmapped on the device
//   kv_vmm_chunk_at(ptr)            the chunk size of the allocation holding ptr
//   kv_vmm_malloc_lazy(size, dev)   a range with nothing mapped yet (restore maps it)
//   kv_vmm_chunk_bytes(device)      the chunk size of allocations of KV_VMM_BIG_MB (64)
//                                   or more; smaller ones use the device granularity
//   kv_vmm_malloc_shadow(size, dev) a range at the device granularity whose chunks
//                                   not mapped by restore read as zeros: every one
//                                   of them maps one shared zeroed chunk of the
//                                   device (read only by convention), so a kernel
//                                   may read the whole range at any time. For
//                                   expert slots that a captured graph addresses
//                                   whether or not they hold an expert
//                                   (kt_kv_hot_lend.py). release maps the zero
//                                   chunk back; restore maps a new zeroed one.
//   kv_vmm_malloc_split(size, dev, device_bytes, numa)
//                                   a range whose chunks from device_bytes on are host
//                                   memory of NUMA node numa (the GPU reads them over
//                                   PCIe); release and restore leave those chunks alone,
//                                   and a released device chunk reads as zeros
//   kv_vmm_place(ptr, device_bytes) a split range's chunks below device_bytes become device memory: each
//                                   host chunk is copied into a new device chunk mapped at the same
//                                   address (the range, its tensors and graphs stay put); returns the
//                                   bytes moved, or -1 when the device has no memory for one
//   kv_vmm_device_bytes(ptr)        how many bytes from the start of ptr's range are
//                                   device memory (its whole size when none is host)
//   kv_vmm_park(device) / kv_vmm_unpark(device)
//                                   every mapped device chunk of the device's ranges to host memory and
//                                   unmapped, and back with its contents
//   KV_VMM_RELEASED_ZERO=1          every range behaves so: a released chunk reads as
//                                   zeros instead of faulting (a diagnostic for stray
//                                   reads of lent KV pages; 10-17)
//
// Build: gcc -O2 -shared -fPIC kv_vmm.c -I$CUDA_HOME/include -L$CUDA_HOME/lib64/stubs -lcuda -o libkv_vmm.so
#include <cuda.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#define MAX_DEV 16
#define MAX_ALLOCS 4096

typedef struct {
  CUdeviceptr va;
  size_t size;  // reserved and mapped size, a multiple of chunk
  size_t chunk;
  int device;
  int nch;
  int shadow;  // unmapped chunks map the device's zero chunk
  int zero;    // unmapped chunks map the zero chunk: shadow ranges, and every range with KV_VMM_RELEASED_ZERO=1
  int host_from;  // chunks from this index on are host memory (nch: none)
  unsigned char* hostm;  // per chunk, 1 = host memory; NULL: host_from decides (kv_vmm_place_range sets it)
  int numa;       // their NUMA node
  CUmemGenericAllocationHandle* h;  // 0 = unmapped (zero: the zero chunk)
  void** park;  // per chunk, its contents in host memory while kv_vmm_park holds it (NULL: none parked)
  void** pbuf;  // per chunk, the pinned host buffer a park copies into: kept from park to park
} alloc_t;

static alloc_t allocs[MAX_ALLOCS];
static int nallocs;
static size_t chunk_of[MAX_DEV];
static size_t released[MAX_DEV];
static CUmemGenericAllocationHandle zero_h[MAX_DEV];  // shadow ranges' zero chunk
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static int ok(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return 1;
  const char* s = NULL;
  cuGetErrorString(r, &s);
  fprintf(stderr, "[kv-vmm] %s failed: %s\n", what, s ? s : "?");
  return 0;
}

static CUmemAllocationProp prop_for(int device) {
  CUmemAllocationProp p = {0};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = device;
  return p;
}

static size_t gran_of[MAX_DEV];

static size_t gran_for(int device) {
  if (gran_of[device]) return gran_of[device];
  CUmemAllocationProp p = prop_for(device);
  size_t gran = 0;
  if (!ok(cuMemGetAllocationGranularity(&gran, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity")) return 0;
  gran_of[device] = gran;
  return gran;
}

static size_t chunk_for(int device) {
  if (chunk_of[device]) return chunk_of[device];
  size_t gran = gran_for(device);
  if (!gran) return 0;
  size_t want = 2u << 20;
  const char* e = getenv("KV_VMM_CHUNK_MB");
  if (e && atol(e) > 0) want = (size_t)atol(e) << 20;
  chunk_of[device] = (want + gran - 1) / gran * gran;
  return chunk_of[device];
}

static int map_handle(CUdeviceptr at, size_t len, CUmemGenericAllocationHandle h, int device) {
  if (!ok(cuMemMap(at, len, 0, h, 0), "cuMemMap")) return 0;
  CUmemAccessDesc d = {0};
  d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  d.location.id = device;
  d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (!ok(cuMemSetAccess(at, len, &d, 1), "cuMemSetAccess")) {
    cuMemUnmap(at, len);
    return 0;
  }
  return 1;
}

// the device's zero chunk (one granule), made and zeroed on first use
static int zero_chunk(int device) {
  if (zero_h[device]) return 1;
  size_t g = gran_for(device);
  CUmemAllocationProp p = prop_for(device);
  CUmemGenericAllocationHandle h;
  CUdeviceptr va;
  if (!g || !ok(cuMemCreate(&h, g, &p, 0), "cuMemCreate (zero chunk)")) return 0;
  if (!ok(cuMemAddressReserve(&va, g, g, 0, 0), "cuMemAddressReserve (zero chunk)")) {
    cuMemRelease(h);
    return 0;
  }
  int good = map_handle(va, g, h, device) && ok(cuMemsetD8(va, 0, g), "cuMemsetD8 (zero chunk)") &&
             ok(cuCtxSynchronize(), "cuCtxSynchronize (zero chunk)");
  cuMemUnmap(va, g);
  cuMemAddressFree(va, g);
  if (!good) {
    cuMemRelease(h);
    return 0;
  }
  zero_h[device] = h;
  return 1;
}

// KV_VMM_RELEASED_ZERO=1: a released chunk of any range reads as zeros instead of
// faulting (the zero granule mapped over it), so a kernel that reads a lent page
// gets zeros, not an illegal address. A diagnostic for stray reads of the KV pool's
// lent pages (10-17: the QSA indexer's illegal access); off by default.
static int released_zero(void) {
  static int v = -1;
  if (v < 0) {
    const char* e = getenv("KV_VMM_RELEASED_ZERO");
    v = e && atoi(e) != 0;
  }
  return v;
}

// the zero granule over every granule of chunk i (or off it)
static int zero_map(alloc_t* a, CUdeviceptr at) {
  size_t g = gran_for(a->device);
  for (size_t o = 0; o < a->chunk; o += g)
    if (!map_handle(at + o, g, zero_h[a->device], a->device)) {
      for (size_t u = 0; u < o; u += g) cuMemUnmap(at + u, g);
      return 0;
    }
  return 1;
}
static void zero_unmap(alloc_t* a, CUdeviceptr at) {
  size_t g = gran_for(a->device);
  for (size_t o = 0; o < a->chunk; o += g) cuMemUnmap(at + o, g);
}

static int is_host(alloc_t* a, int i) { return a->hostm ? a->hostm[i] : i >= a->host_from; }

static int map_chunk(alloc_t* a, int i) {
  CUmemAllocationProp p = prop_for(a->device);
  if (is_host(a, i)) {
    p.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    p.location.id = a->numa;
  }
  CUmemGenericAllocationHandle h;
  if (!ok(cuMemCreate(&h, a->chunk, &p, 0), "cuMemCreate")) return 0;
  CUdeviceptr at = a->va + (size_t)i * a->chunk;
  if (a->zero) zero_unmap(a, at);  // the zero chunk
  if (!map_handle(at, a->chunk, h, a->device)) {
    cuMemRelease(h);
    if (a->zero) zero_map(a, at);
    return 0;
  }
  a->h[i] = h;
  return 1;
}

static void unmap_chunk(alloc_t* a, int i) {
  if (!a->h[i]) return;
  CUdeviceptr at = a->va + (size_t)i * a->chunk;
  cuMemUnmap(at, a->chunk);
  cuMemRelease(a->h[i]);
  a->h[i] = 0;
  if (a->zero) zero_map(a, at);
}

static void* alloc_range(ssize_t size, int device, int lazy, size_t device_bytes, int numa);

void* kv_vmm_malloc(ssize_t size, int device, void* stream) {
  (void)stream;
  return alloc_range(size, device, 0, (size_t)-1, 0);
}

void* kv_vmm_malloc_split(ssize_t size, int device, size_t device_bytes, int numa) {
  return alloc_range(size, device, 0, device_bytes, numa);
}

// a range with no chunk mapped (all counted released): kv_vmm_restore maps
// what is used. For a pool sized past what it needs at once (the SWA pool
// widened for the prefill plan's cap), so the size costs no memory at setup.
void* kv_vmm_malloc_lazy(ssize_t size, int device) { return alloc_range(size, device, 1, (size_t)-1, 0); }

void* kv_vmm_malloc_shadow(ssize_t size, int device) { return alloc_range(size, device, 2, (size_t)-1, 0); }

static void* alloc_range(ssize_t size, int device, int lazy, size_t device_bytes, int numa) {
  if (size <= 0 || device < 0 || device >= MAX_DEV) return NULL;
  cuInit(0);
  pthread_mutex_lock(&mu);
  // the chunk size only for allocations the loan cuts (the K/V buffers, at
  // least KV_VMM_BIG_MB, 64): every other tensor sglang makes in the KV region
  // (Mamba state, req-to-token, a draft's pool) at the device granularity, or
  // each of torch's 2 MiB segments would hold a whole chunk (0.55 GB a card lost
  // at 8 MiB on one setup)
  const char* eb = getenv("KV_VMM_BIG_MB");
  size_t big = (size_t)(eb && atol(eb) > 0 ? atol(eb) : 64) << 20;
  size_t chunk = (size_t)size >= big && lazy != 2 ? chunk_for(device) : gran_for(device);
  // a split range (DeepSeek-V4.1's compressed KV, read a whole prefix at a time) reads its
  // lent pages as zeros instead of faulting, like a shadow range
  int split = device_bytes != (size_t)-1;
  if (!chunk || nallocs >= MAX_ALLOCS || ((lazy == 2 || released_zero() || split) && !zero_chunk(device))) {
    pthread_mutex_unlock(&mu);
    return NULL;
  }
  alloc_t a = {0};
  a.device = device;
  a.chunk = chunk;
  a.shadow = lazy == 2;
  a.zero = a.shadow || released_zero() || split;
  a.size = ((size_t)size + chunk - 1) / chunk * chunk;
  a.nch = (int)(a.size / chunk);
  // the chunk holding byte device_bytes and every later one are host memory
  a.host_from = device_bytes >= a.size ? a.nch : (int)(device_bytes / chunk);
  a.numa = numa;
  a.h = calloc(a.nch, sizeof(*a.h));
  if (!a.h || !ok(cuMemAddressReserve(&a.va, a.size, chunk, 0, 0), "cuMemAddressReserve")) {
    free(a.h);
    pthread_mutex_unlock(&mu);
    return NULL;
  }
  for (int i = 0; i < a.nch && a.zero; i++) {  // the zero chunk first (map_chunk takes it off)
    if (!zero_map(&a, a.va + (size_t)i * chunk)) {
      for (int j = 0; j < i; j++) zero_unmap(&a, a.va + (size_t)j * chunk);
      cuMemAddressFree(a.va, a.size);
      free(a.h);
      pthread_mutex_unlock(&mu);
      return NULL;
    }
  }
  for (int i = 0; i < a.nch && !lazy; i++) {
    if (!map_chunk(&a, i)) {
      for (int j = 0; j < i; j++) unmap_chunk(&a, j);
      cuMemAddressFree(a.va, a.size);
      free(a.h);
      pthread_mutex_unlock(&mu);
      return NULL;
    }
  }
  if (lazy) released[device] += a.size;
  allocs[nallocs++] = a;
  pthread_mutex_unlock(&mu);
  return (void*)a.va;
}

void kv_vmm_free(void* ptr, ssize_t size, int device, void* stream) {
  (void)size; (void)device; (void)stream;
  pthread_mutex_lock(&mu);
  for (int k = 0; k < nallocs; k++) {
    alloc_t* a = &allocs[k];
    if (a->va != (CUdeviceptr)ptr) continue;
    for (int i = 0; i < a->nch; i++) {
      CUdeviceptr at = a->va + (size_t)i * a->chunk;
      if (a->h[i]) {
        cuMemUnmap(at, a->chunk);
        cuMemRelease(a->h[i]);
        a->h[i] = 0;
      } else {
        released[a->device] -= a->chunk;
        if (a->zero) zero_unmap(a, at);  // the zero chunk
      }
    }
    cuMemAddressFree(a->va, a->size);
    if (a->pbuf) {
      for (int i = 0; i < a->nch; i++)
        if (a->pbuf[i]) cuMemFreeHost(a->pbuf[i]);
      free(a->pbuf);
      a->pbuf = NULL;
    }
    if (a->park) {
      free(a->park);
    }
    free(a->h);
    free(a->hostm);
    allocs[k] = allocs[--nallocs];
    break;
  }
  pthread_mutex_unlock(&mu);
}

static alloc_t* find(uintptr_t p) {
  for (int k = 0; k < nallocs; k++)
    if (p >= allocs[k].va && p < allocs[k].va + allocs[k].size) return &allocs[k];
  return NULL;
}

// chunks wholly inside [p + off, p + off + len)
static int range(alloc_t* a, uintptr_t p, size_t off, size_t len, int* i0, int* i1) {
  uintptr_t lo = p + off, hi = p + off + len;
  if (hi > a->va + a->size) hi = a->va + a->size;
  if (hi <= lo) return 0;
  *i0 = (int)((lo - a->va + a->chunk - 1) / a->chunk);
  *i1 = (int)((hi - a->va) / a->chunk);
  return *i1 > *i0;
}

// every chunk that [p + off, p + off + len) touches: what a restore must map
static int range_touched(alloc_t* a, uintptr_t p, size_t off, size_t len, int* i0, int* i1) {
  uintptr_t lo = p + off, hi = p + off + len;
  if (hi > a->va + a->size) hi = a->va + a->size;
  if (hi <= lo) return 0;
  *i0 = (int)((lo - a->va) / a->chunk);
  *i1 = (int)((hi - a->va + a->chunk - 1) / a->chunk);
  if (*i1 > a->nch) *i1 = a->nch;
  return *i1 > *i0;
}

long long kv_vmm_release(void* ptr, size_t off, size_t len) {
  pthread_mutex_lock(&mu);
  alloc_t* a = find((uintptr_t)ptr);
  long long n = 0;
  int i0, i1;
  if (a && range(a, (uintptr_t)ptr, off, len, &i0, &i1)) {
    for (int i = i0; i < i1; i++) {
      if (is_host(a, i)) continue;
      if (!a->h[i]) continue;
      unmap_chunk(a, i);
      n += a->chunk;
    }
    released[a->device] += n;
  }
  pthread_mutex_unlock(&mu);
  return n;
}

long long kv_vmm_restore(void* ptr, size_t off, size_t len) {
  pthread_mutex_lock(&mu);
  alloc_t* a = find((uintptr_t)ptr);
  long long n = 0;
  int i0, i1;
  if (a && range_touched(a, (uintptr_t)ptr, off, len, &i0, &i1)) {
    for (int i = i0; i < i1; i++) {
      if (is_host(a, i)) continue;
      if (a->h[i]) continue;
      if (!map_chunk(a, i)) {
        released[a->device] -= n;
        pthread_mutex_unlock(&mu);
        return -1;
      }
      if (!ok(cuMemsetD8(a->va + (size_t)i * a->chunk, 0, a->chunk), "cuMemsetD8")) {
        released[a->device] -= n;
        pthread_mutex_unlock(&mu);
        return -1;
      }
      n += a->chunk;
    }
    released[a->device] -= n;
  }
  pthread_mutex_unlock(&mu);
  return n;
}

// kv_vmm_park(device): every device chunk mapped under a range of the device goes to host memory (an ordinary
// buffer a chunk) and is unmapped, so the card holds none of them; the ranges, their tensors and the graphs that
// name them stay where they are. kv_vmm_unpark(device) maps each again and copies it back. For a server that
// leaves the cards while another runs (kt_park): the pools that are not token KV (recurrent states, tables) keep
// their contents this way. Returns the bytes moved, or -1 (a park that fails has put back what it took).
long long kv_vmm_unpark(int device);

long long kv_vmm_park(int device) {
  if (device < 0 || device >= MAX_DEV) return -1;
  pthread_mutex_lock(&mu);
  long long n = 0;
  int bad = 0;
  if (!ok(cuCtxSynchronize(), "cuCtxSynchronize (park)")) bad = 1;
  for (int k = 0; k < nallocs && !bad; k++) {
    alloc_t* a = &allocs[k];
    if (a->device != device) continue;
    for (int i = 0; i < a->nch && !bad; i++) {
      if (!a->h[i] || is_host(a, i)) continue;
      if (!a->park) a->park = (void**)calloc((size_t)a->nch, sizeof(void*));
      if (!a->pbuf) a->pbuf = (void**)calloc((size_t)a->nch, sizeof(void*));
      // pinned, and kept for the next park: a pageable buffer made each time took 0.75 s for 1.1 GiB
      void* buf = a->park && a->pbuf ? a->pbuf[i] : NULL;
      if (a->park && a->pbuf && !buf && !ok(cuMemAllocHost(&buf, a->chunk), "cuMemAllocHost (park)")) buf = NULL;
      if (buf) a->pbuf[i] = buf;
      if (!buf || !ok(cuMemcpyDtoH(buf, a->va + (size_t)i * a->chunk, a->chunk), "cuMemcpyDtoH (park)")) {
        bad = 1;
        break;
      }
      a->park[i] = buf;
      unmap_chunk(a, i);
      released[device] += a->chunk;
      n += (long long)a->chunk;
    }
  }
  pthread_mutex_unlock(&mu);
  if (bad) {
    kv_vmm_unpark(device);
    return -1;
  }
  return n;
}

long long kv_vmm_unpark(int device) {
  if (device < 0 || device >= MAX_DEV) return -1;
  pthread_mutex_lock(&mu);
  long long n = 0;
  int bad = 0;
  for (int k = 0; k < nallocs; k++) {
    alloc_t* a = &allocs[k];
    if (a->device != device || !a->park) continue;
    for (int i = 0; i < a->nch; i++) {
      if (!a->park[i]) continue;
      if (!a->h[i]) {
        if (!map_chunk(a, i)) {
          bad = 1;  // its contents stay in host memory for another try
          continue;
        }
        released[device] -= a->chunk;
      }
      if (!ok(cuMemcpyHtoD(a->va + (size_t)i * a->chunk, a->park[i], a->chunk), "cuMemcpyHtoD (unpark)")) {
        bad = 1;
        continue;
      }
      a->park[i] = NULL;  // (the buffer stays in pbuf)
      n += (long long)a->chunk;
    }
  }
  if (!ok(cuCtxSynchronize(), "cuCtxSynchronize (unpark)")) bad = 1;
  pthread_mutex_unlock(&mu);
  return bad ? -1 : n;
}

// the chunk size of the allocation that holds ptr (0: not kv_vmm memory)
size_t kv_vmm_chunk_at(void* ptr) {
  pthread_mutex_lock(&mu);
  alloc_t* a = find((uintptr_t)ptr);
  size_t c = a ? a->chunk : 0;
  pthread_mutex_unlock(&mu);
  return c;
}

size_t kv_vmm_device_bytes(void* ptr) {
  pthread_mutex_lock(&mu);
  alloc_t* a = find((uintptr_t)ptr);
  size_t b = a ? (size_t)a->host_from * a->chunk : 0;
  if (a && a->hostm) {
    int i = 0;
    while (i < a->nch && !a->hostm[i]) i++;
    b = (size_t)i * a->chunk;
  }
  if (a && b > a->size) b = a->size;
  pthread_mutex_unlock(&mu);
  return b;
}

long long kv_vmm_place(void* ptr, size_t device_bytes) {
  pthread_mutex_lock(&mu);
  alloc_t* a = find((uintptr_t)ptr);
  long long n = 0;
  if (!a) {
    pthread_mutex_unlock(&mu);
    return -1;
  }
  int to = device_bytes >= a->size ? a->nch : (int)(device_bytes / a->chunk);
  CUmemAllocationProp p = prop_for(a->device);
  for (int i = a->host_from; i < to; i++) {
    CUdeviceptr at = a->va + (size_t)i * a->chunk;
    if (!a->h[i]) {  // never mapped or released: restore maps it on the device from now on
      a->host_from = i + 1;
      continue;
    }
    CUmemGenericAllocationHandle h;
    CUdeviceptr tmp;
    if (!ok(cuMemCreate(&h, a->chunk, &p, 0), "cuMemCreate (place)")) break;
    if (!ok(cuMemAddressReserve(&tmp, a->chunk, a->chunk, 0, 0), "cuMemAddressReserve (place)")) {
      cuMemRelease(h);
      break;
    }
    int good = map_handle(tmp, a->chunk, h, a->device) && ok(cuMemcpyDtoD(tmp, at, a->chunk), "cuMemcpyDtoD (place)") &&
               ok(cuCtxSynchronize(), "cuCtxSynchronize (place)");
    cuMemUnmap(tmp, a->chunk);
    cuMemAddressFree(tmp, a->chunk);
    if (!good) {
      cuMemRelease(h);
      break;
    }
    cuMemUnmap(at, a->chunk);
    cuMemRelease(a->h[i]);
    a->h[i] = 0;
    if (!map_handle(at, a->chunk, h, a->device)) {  // the host copy is gone: nothing sane to map back
      cuMemRelease(h);
      pthread_mutex_unlock(&mu);
      return -1;
    }
    a->h[i] = h;
    a->host_from = i + 1;
    n += a->chunk;
  }
  pthread_mutex_unlock(&mu);
  return a->host_from >= to ? n : -1;
}

// the reverse of kv_vmm_place: the chunks from device_bytes on become host memory of the range's NUMA node, each
// device chunk copied into a host chunk mapped at the same address; returns the device bytes given back, or -1
long long kv_vmm_unplace(void* ptr, size_t device_bytes) {
  pthread_mutex_lock(&mu);
  alloc_t* a = find((uintptr_t)ptr);
  long long n = 0;
  if (!a) {
    pthread_mutex_unlock(&mu);
    return -1;
  }
  int to = (int)((device_bytes + a->chunk - 1) / a->chunk);
  CUmemAllocationProp p = prop_for(a->device);
  p.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  p.location.id = a->numa;
  for (int i = a->host_from - 1; i >= to; i--) {
    CUdeviceptr at = a->va + (size_t)i * a->chunk;
    CUmemGenericAllocationHandle h;
    CUdeviceptr tmp;
    if (!ok(cuMemCreate(&h, a->chunk, &p, 0), "cuMemCreate (unplace)")) break;
    if (!ok(cuMemAddressReserve(&tmp, a->chunk, a->chunk, 0, 0), "cuMemAddressReserve (unplace)")) {
      cuMemRelease(h);
      break;
    }
    int good = map_handle(tmp, a->chunk, h, a->device);
    if (good && a->h[i])
      good = ok(cuMemcpyDtoD(tmp, at, a->chunk), "cuMemcpyDtoD (unplace)") && ok(cuCtxSynchronize(), "cuCtxSynchronize (unplace)");
    else if (good)
      good = ok(cuMemsetD8(tmp, 0, a->chunk), "cuMemsetD8 (unplace)") && ok(cuCtxSynchronize(), "cuCtxSynchronize (unplace)");
    cuMemUnmap(tmp, a->chunk);
    cuMemAddressFree(tmp, a->chunk);
    if (!good) {
      cuMemRelease(h);
      break;
    }
    if (a->h[i]) {
      cuMemUnmap(at, a->chunk);
      cuMemRelease(a->h[i]);
      a->h[i] = 0;
      n += a->chunk;
    } else {
      if (a->zero) zero_unmap(a, at);
      released[a->device] -= a->chunk;
    }
    if (!map_handle(at, a->chunk, h, a->device)) {
      cuMemRelease(h);
      pthread_mutex_unlock(&mu);
      return -1;
    }
    a->h[i] = h;
    a->host_from = i;
  }
  pthread_mutex_unlock(&mu);
  return n;
}

// the chunks wholly inside [ptr + off, ptr + off + len) to the card (to_device) or to host memory, each copied into
// a new chunk mapped at the same address; returns the bytes moved, or -1 (a chunk could not be made)
// kv_vmm_place_range_live: to host memory, a chunk that is released (lent, nothing mapped) stays as it is: it
// holds nothing to keep, and a host chunk made for it would be zeros copied for no one (a park: the pool's
// free pages are out on the hot lend)
static int place_live_only;

long long kv_vmm_place_range(void* ptr, size_t off, size_t len, int to_device) {
  pthread_mutex_lock(&mu);
  alloc_t* a = find((uintptr_t)ptr);
  long long n = 0;
  int i0, i1;
  // to the card: every chunk the range touches; to host memory: whole chunks only
  if (!a || !(to_device ? range_touched(a, (uintptr_t)ptr, off, len, &i0, &i1)
                        : range(a, (uintptr_t)ptr, off, len, &i0, &i1))) {
    pthread_mutex_unlock(&mu);
    return a ? 0 : -1;
  }
  if (!a->hostm) {
    a->hostm = calloc(a->nch, 1);
    if (!a->hostm) {
      pthread_mutex_unlock(&mu);
      return -1;
    }
    for (int i = 0; i < a->nch; i++) a->hostm[i] = i >= a->host_from;
  }
  for (int i = i0; i < i1; i++) {
    if (a->hostm[i] == !to_device) continue;
    if (!to_device && place_live_only && !a->h[i]) continue;
    CUmemAllocationProp p = prop_for(a->device);
    if (!to_device) {
      p.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      p.location.id = a->numa;
    }
    CUdeviceptr at = a->va + (size_t)i * a->chunk;
    CUmemGenericAllocationHandle h;
    CUdeviceptr tmp;
    if (!ok(cuMemCreate(&h, a->chunk, &p, 0), "cuMemCreate (place range)")) break;
    if (!ok(cuMemAddressReserve(&tmp, a->chunk, a->chunk, 0, 0), "cuMemAddressReserve (place range)")) {
      cuMemRelease(h);
      break;
    }
    int good = map_handle(tmp, a->chunk, h, a->device);
    if (good && a->h[i])
      good = ok(cuMemcpyDtoD(tmp, at, a->chunk), "cuMemcpyDtoD (place range)") && ok(cuCtxSynchronize(), "sync");
    else if (good)
      good = ok(cuMemsetD8(tmp, 0, a->chunk), "cuMemsetD8 (place range)") && ok(cuCtxSynchronize(), "sync");
    cuMemUnmap(tmp, a->chunk);
    cuMemAddressFree(tmp, a->chunk);
    if (!good) {
      cuMemRelease(h);
      break;
    }
    if (a->h[i]) {
      cuMemUnmap(at, a->chunk);
      cuMemRelease(a->h[i]);
      a->h[i] = 0;
    } else {
      if (a->zero) zero_unmap(a, at);
      released[a->device] -= a->chunk;
    }
    if (!map_handle(at, a->chunk, h, a->device)) {
      cuMemRelease(h);
      pthread_mutex_unlock(&mu);
      return -1;
    }
    a->h[i] = h;
    a->hostm[i] = !to_device;
    n += a->chunk;
  }
  pthread_mutex_unlock(&mu);
  return n;
}

long long kv_vmm_place_range_live(void* ptr, size_t off, size_t len, int to_device) {
  place_live_only = 1;
  long long n = kv_vmm_place_range(ptr, off, len, to_device);
  place_live_only = 0;
  return n;
}

size_t kv_vmm_released_bytes(int device) { return device >= 0 && device < MAX_DEV ? released[device] : 0; }

size_t kv_vmm_chunk_bytes(int device) {
  pthread_mutex_lock(&mu);
  size_t c = device >= 0 && device < MAX_DEV ? chunk_for(device) : 0;
  pthread_mutex_unlock(&mu);
  return c;
}

// this library's kv_vmm_restore maps every chunk a range touches
int kv_vmm_restore_touches(void) { return 1; }

// device memory mapped under every range of a device (host chunks apart), the zero chunk included
size_t kv_vmm_mapped_bytes(int device) {
  size_t n = 0;
  if (device < 0 || device >= MAX_DEV) return 0;
  pthread_mutex_lock(&mu);
  for (int k = 0; k < nallocs; k++) {
    alloc_t* a = &allocs[k];
    if (a->device != device) continue;
    for (int i = 0; i < a->nch; i++)
      if (a->h[i] && !is_host(a, i)) n += a->chunk;
  }
  if (zero_h[device]) n += gran_of[device];
  pthread_mutex_unlock(&mu);
  return n;
}

// virtual bytes reserved by the ranges of a device
size_t kv_vmm_virtual_bytes(int device) {
  size_t n = 0;
  if (device < 0 || device >= MAX_DEV) return 0;
  pthread_mutex_lock(&mu);
  for (int k = 0; k < nallocs; k++)
    if (allocs[k].device == device) n += allocs[k].size;
  pthread_mutex_unlock(&mu);
  return n;
}

// this library's kv_vmm_place_range brings every touched chunk to the card
int kv_vmm_place_touches(void) { return 1; }
