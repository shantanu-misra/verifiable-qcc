#!/usr/bin/env bash
# Assemble a Kaggle submission zip from chosen circuits + the untouched dfthcsos baseline.
#
#   scripts/bundle.sh circuits/invsqrt/best.kmb circuits/dqi/best.kmb [out.zip]
#
# Pass the literal word "baseline" for either circuit to use data/sample_submission/<task>.kmb.
# Zip layout required by the competition (Overview > Submission File):
#   invsqrt.kmb|kmx, dqi.kmb|kmx, dfthcsos/{angle_k_0..9.kmb, submission.npz}
set -euo pipefail
cd "$(dirname "$0")/.."

invsqrt="${1:?invsqrt circuit or 'baseline'}"
dqi="${2:?dqi circuit or 'baseline'}"
out="${3:-submissions/submission_$(date -u +%Y%m%dT%H%M%SZ).zip}"

[ "$invsqrt" = baseline ] && invsqrt=data/sample_submission/invsqrt.kmb
[ "$dqi" = baseline ] && dqi=data/sample_submission/dqi.kmb

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp "$invsqrt" "$stage/invsqrt.${invsqrt##*.}"
cp "$dqi" "$stage/dqi.${dqi##*.}"
cp -R data/sample_submission/dfthcsos "$stage/dfthcsos"

mkdir -p "$(dirname "$out")"
(cd "$stage" && zip -qr "$OLDPWD/$out" .)
echo "wrote $out"
unzip -l "$out"
{
  echo "created_utc: $(date -u +%FT%TZ)"
  echo "invsqrt: $invsqrt  sha256: $(shasum -a 256 "$invsqrt" | cut -d' ' -f1)"
  echo "dqi:     $dqi  sha256: $(shasum -a 256 "$dqi" | cut -d' ' -f1)"
  echo "dfthcsos: baseline (data/sample_submission/dfthcsos)"
  echo "kaggle_public_score: TODO fill in after upload"
} > "${out%.zip}.txt"
echo "sidecar: ${out%.zip}.txt"
