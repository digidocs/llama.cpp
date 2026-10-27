#!/bin/bash
# decode A/B with llama-bench, configs alternated round by round so drift (thermals, DRAM refresh, page cache) hits all
# usage: [N=128] [R=3] ab_bench.sh <rounds> <label>=<worktree>[:VAR=val,VAR=val][@arg,arg] ...   (@: extra llama-bench arguments)
# one model load per config per round; prints "round label tg t/s +- sd" and a mean per config at the end
. "$(dirname "$0")/common.sh"
N=${N:-128}; R=${R:-3}; rounds=$1; shift
declare -A sum cnt
for ((i = 1; i <= rounds; i++)); do
    for c in "$@"; do
        label=${c%%=*}; rest=${c#*=}; args=""
        [[ $rest == *@* ]] && args=${rest#*@} && rest=${rest%%@*}
        dir=${rest%%:*}; envs=""
        [[ $rest == *:* ]] && envs=${rest#*:}
        res=$(env ${envs//,/ } $dir/build/bin/llama-bench -m $M $OFFLOAD $CPU -lm dio -p 0 -n $N -r $R ${args//,/ } 2>/dev/null \
            | grep "tg$N" | awk -F'|' '{print $(NF-1)}' | xargs)
        echo "$i $label $res"
        tg=${res%% *}
        [ -n "$tg" ] && sum[$label]=$(awk -v a=${sum[$label]:-0} -v b=$tg 'BEGIN{print a+b}') && cnt[$label]=$(( ${cnt[$label]:-0} + 1 ))
    done
done
for c in "$@"; do
    label=${c%%=*}
    [ -n "${cnt[$label]}" ] && printf "mean %-16s %.2f t/s over %d runs\n" "$label" "$(awk -v a=${sum[$label]} -v b=${cnt[$label]} 'BEGIN{print a/b}')" "${cnt[$label]}"
done
