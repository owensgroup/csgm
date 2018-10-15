### csgm

CUDA seeded graph matchin implementation.

Currently, only tested on undirected, unweighted graphs.

Does not produce the _exact_ same results as Python version due to floating point issues.  (Though this only happens if the `spmm_convex_combination` gets called, otherwise entries are integer valued.)