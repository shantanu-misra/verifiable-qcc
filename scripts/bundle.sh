#!/usr/bin/env bash
# Assemble a Kaggle submission zip from chosen circuits + the untouched dfthcsos baseline.
#
#   scripts/bundle.sh circuits/invsqrt/best.kmb circuits/dqi/best.kmb [circuits/dfthcsos/<dir>|baseline] [out.zip]
#
# Pass the literal word "baseline" for any task to use data/sample_submission/<task>; the dfthcsos
# argument is a directory holding angle_k_0..K-1.kmb + submission.npz (default: baseline).
# Zip layout required by the competition (Overview > Submission File):
#   invsqrt.kmb|kmx, dqi.kmb|kmx, dfthcsos/{angle_k_0..9.kmb, submission.npz}
set -euo pipefail
cd "$(dirname "$0")/.."

invsqrt="${1:?invsqrt circuit or 'baseline'}"
dqi="${2:?dqi circuit or 'baseline'}"
dfthcsos=baseline
out=submissions/submission.zip
case "${3:-}" in
  "") ;;
  *.zip) out="$3" ;;
  *) dfthcsos="$3"; out="${4:-$out}" ;;
esac
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
side="${out%.zip}_${stamp}.txt"   # one record per bundle, even though the zip name is reused

[ "$invsqrt" = baseline ] && invsqrt=data/sample_submission/invsqrt.kmb
[ "$dqi" = baseline ] && dqi=data/sample_submission/dqi.kmb
[ "$dfthcsos" = baseline ] && dfthcsos=data/sample_submission/dfthcsos
[ -f "$dfthcsos/angle_k_0.kmb" ] || [ -f "$dfthcsos/angle_k_0.kmx" ] || { echo "no angle_k_0 in $dfthcsos" >&2; exit 2; }
[ -f "$dfthcsos/submission.npz" ] || { echo "no submission.npz in $dfthcsos" >&2; exit 2; }

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp "$invsqrt" "$stage/invsqrt.${invsqrt##*.}"
cp "$dqi" "$stage/dqi.${dqi##*.}"
mkdir "$stage/dfthcsos" && cp "$dfthcsos"/angle_k_*.km[bx] "$dfthcsos"/submission.npz "$stage/dfthcsos/"

mkdir -p "$(dirname "$out")"
(cd "$stage" && zip -qr "$OLDPWD/$out" .)
echo "wrote $out"
unzip -l "$out"
{
  echo "created_utc: $(date -u +%FT%TZ)"
  echo "zip: $out  sha256: $(shasum -a 256 "$out" | cut -d' ' -f1)"
  echo "invsqrt: $invsqrt  sha256: $(shasum -a 256 "$invsqrt" | cut -d' ' -f1)"
  echo "dqi:     $dqi  sha256: $(shasum -a 256 "$dqi" | cut -d' ' -f1)"
  echo "dfthcsos: $dfthcsos  sha256(angle_k_*,submission.npz): $(cat "$dfthcsos"/angle_k_*.km[bx] "$dfthcsos"/submission.npz | shasum -a 256 | cut -d' ' -f1)"
  echo "kaggle_public_score: TODO fill in after upload"
} > "$side"
echo "sidecar: $side"
