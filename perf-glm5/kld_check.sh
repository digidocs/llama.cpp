#!/bin/bash
# model-level output check: 2 chunks of 512 tokens; default UB=1 runs each token in its own ubatch (decode graphs), UB=512 the prefill graphs
# usage: kld_check.sh base <out.kld>              save reference logits (EXTRA="..." adds arguments)
#        kld_check.sh cmp  <ref.kld> [label]       compare the current build/env against a reference
cd ~/llama.cpp-glm5 && . perf-glm5/common.sh
ARGS="-m $M -f perf-glm5/kld_text.txt -c 512 -b 512 -ub ${UB:-1} --chunks 2 $OFFLOAD $CPU -lm ${LOAD_MODE:-dio} ${EXTRA}"
if [ "$1" = base ]; then
    ./build/bin/llama-perplexity $ARGS --kl-divergence-base "$2" 2>&1 | grep -aE "Final estimate|error" | tail -2
else
    ./build/bin/llama-perplexity $ARGS --kl-divergence-base "$2" --kl-divergence 2>&1 | grep -aE "Mean +KLD|Maximum KLD|Same top p|RMS Δp|Mean +PPL\(Q\)|ln\(PPL|error" | sed "s/^/${3:-cmp}: /"
fi
