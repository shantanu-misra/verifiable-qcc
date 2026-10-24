#!/usr/bin/env bash
# Run the OFFICIAL competition verifier on one circuit, inside the Docker image.
#
#   scripts/verify.sh invsqrt circuits/invsqrt/foo.kmx [extra verifier args]
#   scripts/verify.sh dqi     circuits/dqi/foo.kmb     [extra verifier args]
#
# Prints the verifier's JSON (status/score/...) to stdout and also saves it to
# runs/<task>_<basename>_<utc-timestamp>.json. Exit code is the verifier's (1 on FAIL).
# Never edit data/verifiers/; this script only wraps it.
set -euo pipefail
cd "$(dirname "$0")/.."

task="${1:?task: invsqrt | dqi}"; shift
circuit="${1:?path to .kmx/.kmb}"; shift

case "$task" in
  invsqrt) verifier=data/verifiers/invsqrt/verify_submission_invsqrt.py ;;
  dqi)     verifier=data/verifiers/dqi/verify_submission_dqi.py ;;
  *) echo "unknown task: $task (dfthcsos is submitted as baseline; use data/verifiers/dfthcsos/verify.py directly)" >&2; exit 2 ;;
esac

mkdir -p runs
out="runs/${task}_$(basename "${circuit%.*}")_$(date -u +%Y%m%dT%H%M%SZ).json"
set +e
docker compose run --rm -T dev python "$verifier" "$circuit" "$@" | tee "$out"
rc=${PIPESTATUS[0]}
set -e
echo "saved: $out" >&2
exit "$rc"
