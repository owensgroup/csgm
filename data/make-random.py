#!/usr/bin/env python

"""
    random-mtx.py
"""

import sys
import argparse
import numpy as np
from scipy import sparse
from scipy.io import mmwrite

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=4096)
    parser.add_argument('--num-seeds', type=int, default=100)
    parser.add_argument('--density', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Random A
    A = sparse.random(args.dim, args.dim, args.density)
    A = ((A + A.T) > 0).astype(int)
    A.setdiag(0)
    A.eliminate_zeros()
    
    # Randomly permute A to make B
    p_data = np.ones(args.dim, dtype=int)
    p_rows = np.arange(args.dim, dtype=int)
    p_cols = np.arange(args.dim, dtype=int)
    p_cols[args.num_seeds:] = np.random.permutation(p_cols[args.num_seeds:])
    
    P = sparse.csr_matrix((p_data, (p_rows, p_cols)))
    B = P.dot(A).dot(P.T)
    
    mmwrite('data/A', A, symmetry='symmetric')
    mmwrite('data/B', B, symmetry='symmetric')
