#!/bin/bash
# How long until my jobs run? Answers three questions the scheduler will not volunteer:
# where each job sits in the queue, how many jobs outrank it, and what the backfill
# scheduler currently estimates.
#
#   bash hpc/queue.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"

echo "== my jobs"
squeue -u "${USER}" --start -o "%.10i %.10P %.12j %.8T %.20S %.12r" || true

echo
echo "== how many pending jobs outrank each of mine"
# %F is the array job's base id: "%i" yields 18291306_[1-4], which squeue -j rejects.
for jid in $(squeue -h -u "${USER}" -t PD -o "%F" | sort -u); do
    part=$(squeue -h -j "${jid}" -o "%P" | head -1)
    mine=$(squeue -h -j "${jid}" -o "%Q" | head -1)
    [ -z "${mine}" ] && continue
    ahead=$(squeue -h -p "${part}" -t PD -o "%Q" | awk -v m="${mine}" '$1 > m' | wc -l)
    printf "  job %-10s partition %-10s priority %-10s  %s pending jobs ahead\n" \
        "${jid}" "${part}" "${mine}" "${ahead}"
done

echo
echo "== free GPUs right now, by type"
for node in $(sinfo -h -N -p "${ACRC_GPU_PARTITION}" -o "%N" | sort -u); do
    info=$(scontrol show node "${node}" 2>/dev/null) || continue
    cfg=$(echo "${info}" | grep -o "CfgTRES=.*" | grep -o "gres/gpu=[0-9]*" | cut -d= -f2)
    alloc=$(echo "${info}" | grep -o "AllocTRES=.*" | grep -o "gres/gpu=[0-9]*" | cut -d= -f2)
    kind=$(echo "${info}" | grep -o "Gres=gpu:[^(]*" | cut -d: -f2)
    cores_free=$(echo "${info}" | awk -F'CPUAlloc=' 'NF>1{print $2}' | awk '{print $1}')
    [ -z "${cfg}" ] && continue
    [ -z "${alloc}" ] && alloc=0
    printf "  %-12s %-10s %s of %s GPUs busy, %s cores allocated\n" \
        "${node}" "${kind}" "${alloc}" "${cfg}" "${cores_free:-?}"
done

echo
echo "A start time of Unknown or N/A means the backfill scheduler has not placed the job"
echo "yet; it recomputes every few minutes. Estimates assume every running job uses its"
echo "full requested time, so jobs usually start earlier than predicted."
