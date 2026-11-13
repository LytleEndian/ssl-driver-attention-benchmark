#!/usr/bin/env bash
# extract_bdda_frames.sh - BDD-A frame extraction, identical ffmpeg settings to
# SCOUT's scripts/extract_BDD-A_frames.sh, run in parallel and with logging.
#
#   camera:  ffmpeg -r 29 (output rate)  -> resampled to the gaze-map clock, JPEG q:v 2, %05d from 0
#   gazemap: crop=1024:576:0:91 (SCOUT's crop of the 1024x768 map to the camera field), PNG, -r 29
#
# Output goes into the dataset tree where SCOUT's dataloader expects it:
#   $BDDA_PATH/<split>/camera_frames/<vid>/00000.jpg
#   $BDDA_PATH/<split>/gazemap_frames/<vid>/00000.png
#
# Differences from SCOUT's script (behaviour only, not output):
#   - camera videos without a gaze-map video are skipped instead of erroring
#   - a camera video ffmpeg cannot read (e.g. training/922) is logged and its dirs removed
#   - runs JOBS videos at once; frame-count mismatches are logged per video
#
# usage: BDDA_PATH=/srv/datasets/BDDA JOBS=8 bash tools/extract_bdda_frames.sh
set -euo pipefail
export BDDA_PATH=${BDDA_PATH:-/srv/datasets/BDDA}
JOBS=${JOBS:-8}
LOG=${LOG:-meta/extract_bdda.log}
mkdir -p "$(dirname "$LOG")"

one() {
  local set_dir=$1 video=$2 fname gz cdir gdir nc ng
  fname=$(basename "$video" .mp4)
  gz="$BDDA_PATH/$set_dir/gazemap_videos/${fname}_pure_hm.mp4"
  if [ ! -f "$gz" ]; then echo "SKIP  $set_dir $fname no-gazemap"; return 0; fi
  cdir="$BDDA_PATH/$set_dir/camera_frames/$fname"
  gdir="$BDDA_PATH/$set_dir/gazemap_frames/$fname"
  rm -rf "$cdir" "$gdir"; mkdir -p "$cdir" "$gdir"
  if ! ffmpeg -nostdin -loglevel error -y -i "$video" -start_number 0 -f image2 -q:v 2 -r 29 "$cdir/%05d.jpg"; then
    echo "FAIL  $set_dir $fname camera"; rm -rf "$cdir" "$gdir"; return 0
  fi
  if ! ffmpeg -nostdin -loglevel error -y -i "$gz" -filter:v "crop=1024:576:0:91" -start_number 0 -f image2 -r 29 "$gdir/%05d.png"; then
    echo "FAIL  $set_dir $fname gazemap"; rm -rf "$cdir" "$gdir"; return 0
  fi
  nc=$(find "$cdir" -name '*.jpg' | wc -l); ng=$(find "$gdir" -name '*.png' | wc -l)
  if [ "$nc" -ne "$ng" ]; then echo "COUNT $set_dir $fname camera=$nc gaze=$ng"; else echo "OK    $set_dir $fname frames=$nc"; fi
}
export -f one

for s in training validation test; do
  for v in "$BDDA_PATH/$s"/camera_videos/*.mp4; do printf '%s\0%s\0' "$s" "$v"; done
done | xargs -0 -n2 -P "$JOBS" bash -c 'one "$@"' _ 2>&1 | tee "$LOG"

echo "---- summary"
awk '{print $1}' "$LOG" | sort | uniq -c
