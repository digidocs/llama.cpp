#!/bin/bash
# layer 3 experts back to the CPU (not bit-exact vs the reference): context 280K with production MTP settings
# logs MemAvailable / VRAM every 5 s to l3cpu-mem.txt
CTX=${CTX:-280000}
( while true; do echo "$(date +%T) avail_mib=$(( $(grep MemAvailable /proc/meminfo | awk "{print \$2}") / 1024 )) vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 5; done ) > ~/glm5-work/l3cpu-mem.txt &
MON=$!
A="-ctkd q8_0 -ctvd q8_0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.9 --spec-draft-n-ubatch 256 -td 24 -Crd 0-23 --cpu-strict-draft 1"
GPU_EXPS="blk\.999\.ffn_none=CUDA0" SKIP_PP=1 DEEP=${DEEP:-940000} CTX=$CTX MAX_TOKENS=400 $(dirname "$0")/spec_run.sh l3cpu-c$CTX $A
kill $MON
echo "min avail_mib: $(grep -o "avail_mib=[0-9]*" ~/glm5-work/l3cpu-mem.txt | cut -d= -f2 | sort -n | head -1)  max vram_mib: $(grep -o "vram_mib=[0-9]*" ~/glm5-work/l3cpu-mem.txt | cut -d= -f2 | sort -n | tail -1)"
