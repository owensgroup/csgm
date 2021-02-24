#!/bin/bash

# benchmark.sh

make clean ; make

# --
# kasios

for NUM_NODES in 1024 2048 4096 8192 16384 32768; do
  ./csgm \
    --A data/kasios/calls_A_${NUM_NODES}_100.mtx \
    --B data/kasios/calls_B_${NUM_NODES}_100.mtx \
    --num-seeds 100
done

# num_nodes=1024  | final_dist=0   | time_ms=127.032
# num_nodes=2048  | final_dist=4   | time_ms=222.967
# num_nodes=4096  | final_dist=4   | time_ms=649.221
# num_nodes=8192  | final_dist=8   | time_ms=2012.33
# num_nodes=16384 | final_dist=16  | time_ms=2903.21
# num_nodes=32768 | final_dist=OOM | time_ms=OOM

# --
# connectome (eps=1)

function run_10 {
  PROB=$1
  NUM_SEEDS=$2
  
  ./csgm \
    --A data/connectome/$PROB/sparse/A.mtx \
    --B data/connectome/$PROB/sparse/B.mtx \
    --num-seeds $NUM_SEEDS                 \
    --auction-max-eps 1.0 \
    --auction-min-eps 1.0
}

run_10 DS00833 833
run_10 DS01216 1216
run_10 DS01876 1876
run_10 DS03231 3231
run_10 DS06481 6481
run_10 DS16784 16784



function run_05 {
  PROB=$1
  NUM_SEEDS=$2
  
  ./csgm \
    --A data/connectome/$PROB/sparse/A.mtx \
    --B data/connectome/$PROB/sparse/B.mtx \
    --num-seeds $NUM_SEEDS                 \
    --auction-max-eps 0.5 \
    --auction-min-eps 0.5
}

run_05 DS00833 833
run_05 DS01216 1216
run_05 DS01876 1876
run_05 DS03231 3231
run_05 DS06481 6481
run_05 DS16784 16784

Connectome
  CSGM
    DS00833 | eps=1.0 | final_dist=11538  | time_ms=139.469
    DS01216 | eps=1.0 | final_dist=19360  | time_ms=202.842
    DS01876 | eps=1.0 | final_dist=36936  | time_ms=502.3
    DS03231 | eps=1.0 | final_dist=73746  | time_ms=1863.96
    DS06481 | eps=1.0 | final_dist=184832 | time_ms=5336.96
    DS16784 | eps=1.0 | final_dist=596370 | time_ms=24495.4
    DS00833 | eps=0.5 | final_dist=11466  | time_ms=278.326 
    DS01216 | eps=0.5 | final_dist=19288  | time_ms=568.548 
    DS01876 | eps=0.5 | final_dist=36764  | time_ms=781.799 
    DS03231 | eps=0.5 | final_dist=73346  | time_ms=3775.97 
    DS06481 | eps=0.5 | final_dist=183796 | time_ms=15250.7 
    DS16784 | eps=0.5 | final_dist=592822 | time_ms=69727.3 