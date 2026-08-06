# Resume-aware training loop, sourced by the Stage 1/3/4 job scripts.
#
# Every trainer here saves its optimiser state and step count to --ckpt after each chunk and
# resumes from it on the next call, so a walltime limit costs nothing: the job keeps calling
# the trainer until the target step count is reached, and if the allocation runs out first,
# resubmitting the same job continues from the saved step.

ckpt_step () {
    python - "$1" <<'PY'
import os, sys, torch
p = sys.argv[1]
print(torch.load(p, map_location="cpu")["step"] if os.path.exists(p) else 0)
PY
}

# train_until <target> <ckpt> <chunk_seconds> <job_seconds> <command...>
train_until () {
    local target="$1" ckpt="$2" chunk="$3" job_seconds="$4"
    shift 4
    local deadline=$(( $(date +%s) + job_seconds - chunk - 300 ))
    local step
    step=$(ckpt_step "${ckpt}")
    echo "== ${ckpt}: at step ${step}/${target}"

    while [ "${step}" -lt "${target}" ]; do
        if [ "$(date +%s)" -ge "${deadline}" ]; then
            echo "== out of walltime at step ${step}/${target}."
            echo "== resubmit this job to continue from the checkpoint; nothing is lost."
            return 0
        fi
        "$@" --ckpt "${ckpt}" --target "${target}" --wall "${chunk}"
        local prev="${step}"
        step=$(ckpt_step "${ckpt}")
        if [ "${step}" -le "${prev}" ]; then
            echo "== step did not advance (${prev} -> ${step}); stopping to avoid a spin loop." >&2
            return 1
        fi
    done
    echo "== ${ckpt}: reached ${step}/${target}"
}
