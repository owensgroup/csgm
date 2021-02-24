#!/bin/bash

# run.sh

# cd data
# python make-data.py
# cd ..

make clean ; make

# python data/make-random.py --num-seeds 100 --dim 8192
# ./csgm --A data/A.mtx --B data/B.mtx --num-seeds 100 --sgm-debug 1

for NUM_NODES in 1024 2048 4096 8192 16384 32768 4096; do
  ./csgm \
    --A data/kasios/calls_A_${NUM_NODES}_100.mtx \
    --B data/kasios/calls_B_${NUM_NODES}_100.mtx \
    --num-seeds 100
done