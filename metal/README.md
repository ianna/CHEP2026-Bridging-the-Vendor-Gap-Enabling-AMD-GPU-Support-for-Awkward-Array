<img width="1472" height="1448" alt="image" src="https://github.com/user-attachments/assets/448166d6-0b8e-40c2-b7f5-281f72385517" />

A few things worth noting visually now that all three backends are on the same axes:
The CPU lines (blue for Mac, green for della-milan) share a characteristic V-shape from lower-left to upper-right — both accelerate as data outgrows cache. The GPU lines run roughly horizontal because they're bottlenecked by fixed launch overhead rather than data volume. That's what produces the crossover: the CPU slope is steep, the GPU slope is shallow, and they intersect somewhere between 64 K and 1 M elements.
The i64 gpu_rnd dot at 1 M / 16 K (885 µs, off the top of the f32 chart's intuitive range) tells the PCIe story clearly — 8 MB of doubles crossing the bus dominates everything else at that size.

<img width="1472" height="1740" alt="image" src="https://github.com/user-attachments/assets/0afa8cb2-3ffd-4fa1-9d1e-8ee452e4fb72" />

Three patterns stand out clearly.

**f32: GPU wins only above the command-buffer floor.** The Metal command buffer has a fixed ~97 µs overhead regardless of data size. At 1 K elements the CPU finishes in 269 *nanoseconds* — 370× faster than any Metal path. At 64 K the CPU is still 5× faster. Only at 1 M elements does the GPU pull ahead: dispatch-only is 2.45× faster than CPU, and even the unified roundtrip (which still pays a 4 MB memcpy) is 1.6× faster. The alloc roundtrip crosses back below CPU at 1 M — the three `newBuffer` API calls cost ~100 µs each regardless of data size.

**i64: CPU wins at every tested size.** The CPU scalar loop runs at 12.6 Gelem/s for i64 (likely LLVM auto-vectorizing to NEON 128-bit integer SIMD, two i64 per cycle). The GPU has no i64 SIMD lane — each shader thread accumulates serially with integer ALUs that are narrower and slower than the CPU's vector unit. At 1 M elements: CPU 83 µs vs Metal 120 µs. The crossover for i64 would require much larger arrays than tested here, if it exists at all on this chip.

**The allocation tax is the dominant cost at large sizes.** The gap between `metal_alloc` (464 µs) and `metal_unified` (162 µs) at 1 M is 302 µs — three `newBuffer` calls at ~100 µs each. The data write itself (4 MB memcpy into unified memory) costs only ~56 µs at ~71 GB/s. So for any caller running the reduction repeatedly on a fixed data shape, persistent shared buffers are essential; otherwise the API overhead swamps the compute gain.

<img width="1472" height="1782" alt="image" src="https://github.com/user-attachments/assets/5a35e17f-e714-4036-9f0e-8e77cddffa62" />

The short version for the codebase: a dispatcher sitting in front of the reduce-sum kernels should route on three things in order — dtype first, then element count, then whether a persistent buffer is already live. For f32 with a pre-warmed buffer above ~400 K elements, send it to Metal. For i64 at any size you're likely to see in an awkward array (up to at least 1–2 M), stay on CPU. And if a caller would need to allocate a fresh Metal buffer just for this one call, the CPU is faster regardless of size — the `newBuffer` tax alone exceeds what the CPU takes to finish the reduction.
