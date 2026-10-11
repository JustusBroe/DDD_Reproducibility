#!/usr/bin/env bash
# Write one self-contained UCloud batch script per shard.
#   bash fig11a/make_batch_scripts.sh 8
# Upload fig11a/batch_jobs/job_00.sh ... job_07.sh as eight batch jobs.
# Each job: Ubuntu image, 1x NVIDIA B200, same project folder mounted.
set -eo pipefail

K="${1:?usage: bash make_batch_scripts.sh NUM_SHARDS}"
case "$K" in
    ''|*[!0-9]*) echo "NUM_SHARDS must be a positive integer" >&2; exit 1 ;;
esac
if [ "$K" -lt 1 ]; then
    echo "NUM_SHARDS must be >= 1" >&2
    exit 1
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/batch_jobs"
mkdir -p "$OUT"
# Drop scripts from a previous split so stale shards are not uploaded by mistake.
rm -f "$OUT"/job_*.sh

for ((i = 0; i < K; i++)); do
    dest=$(printf "%s/job_%02d.sh" "$OUT" "$i")
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            SHARD=*) line="SHARD=\"${i}\"" ;;
            NUM_SHARDS=*) line="NUM_SHARDS=\"${K}\"" ;;
        esac
        printf '%s\n' "$line"
    done < "$DIR/batch_job.sh" > "$dest"
    chmod +x "$dest"
    echo "wrote $dest"
done
