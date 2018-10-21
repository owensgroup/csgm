### csgm

CUDA seeded graph matching (SGM) implementation.

Currently, only tested on undirected, unweighted graphs.

#### Usage

```
git clone --recursive https://github.com/owensgroup/csgm
```

See `run.sh` for usage.

#### Todo
- [=] Profiling
- [=] Test on connectome graphs
- [ ] Performance testing on kasios graphs (esp. compared to fused implementation)
- [=] Re-implement auction algorithm in CUB
