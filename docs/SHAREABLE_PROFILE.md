# VFD Proof Dashboard - Performance Profile

## Quick Summary

| Metric | Before Phase 1.5 | After Phase 1.5 | Improvement |
|--------|-----------------|-----------------|-------------|
| Total runtime (dim=600) | 193 seconds | 6.6 seconds | **29x faster** |
| compute_invariants | 185 seconds | 0.56 seconds | **330x faster** |
| Peak memory | 507 MB | 151 MB | **3.4x less** |

## Test Configuration

- **internal_dim**: 600 (50 orbits × 12 orbit_size)
- **cell_count**: 16
- **propagation_range**: 1
- **stability**: off (default)

## Key Optimizations

### 1. Probe-Based Kernel Verification

Replaced dense matrix operations with sparse probe-based testing:

```
Before: K = K.toarray()  # 9600×9600 dense matrix
        error = norm(K - K.conj().T)  # O(n²) memory + compute

After:  for _ in range(32):  # 32 probe pairs
            Kv = K @ v  # Sparse matrix-vector: O(nnz)
            error = max(error, |<u,Kv> - <Ku,v>|)
```

### 2. Fast Mode D3 Check

Default `fast_mode=True` skips expensive eigenvalue computation:
- Random sampling provides confidence bound
- Explicit eigenvalue check available via `--stability on`

### 3. Stability Off by Default

`--stability off` is now the default, saving ~300MB memory.

### 4. Bridge Caching

First 100 zeta zeros embedded in code:
- Instant startup (no mpmath computation)
- Disk cache for larger counts

### 5. Incremental Sweep Writes

Sweep results written after each grid point:
- Resumable if interrupted
- Partial results in `sweep_results_partial.json`

## Commands

```bash
# Standard run (fast, ~7s)
rhdiag run --internal-dim 600

# With performance monitoring
rhdiag run --internal-dim 600 --perf --trace

# With stability analysis (slower)
rhdiag run --internal-dim 600 --stability on

# Parameter sweep with incremental saves
rhdiag sweep --param1 cell_count --values1 8,16,24 \
             --param2 propagation_range --values2 1,2,3
```

## Benchmark Results

```
Step                 | Duration | Memory
---------------------|----------|--------
cli_init             | 0.0005s  | 118 MB
build_state          | 6.57s    | 151 MB
  create_vfdspace    | 0.0001s  | 139 MB
  create_operators   | 0.01s    | 139 MB
  create_kernel      | 0.62s    | 151 MB
  compute_invariants | 0.56s    | 151 MB
  compute_spectrum   | 0.0004s  | 151 MB
closure_ladder       | 0.0002s  | 151 MB
---------------------|----------|--------
TOTAL                | 6.62s    | 151 MB (peak)
```

## Environment

- Python 3.8.10
- NumPy 1.24.4, SciPy 1.10.1
- Platform: Linux (WSL2)
- RAM: 32 GB (28 GB available)

## Files Changed

- `src/vfd_dash/vfd/kernels.py` - Probe-based D1/D2 verification
- `src/vfd_dash/cli.py` - `--stability`, `--perf`, `--trace` flags
- `src/vfd_dash/sweep.py` - Incremental writes
- `src/vfd_dash/bridge/reference_data.py` - Embedded zeros cache
- `src/vfd_dash/diagnostics/perf.py` - Performance monitoring

## Tests

All 142 tests pass including new performance tests:
- `test_weyl_fast_mode_runtime_smoke` - Verifies probe-based is fast
- `test_sweep_resumable_writes_partial` - Verifies incremental saves
- `test_bridge_uses_cached_zeros` - Verifies embedded zeros work
