#!/usr/bin/env bash
# Download the Replica data for the 8 OpenLex3D scenes into data/ (or $DATA_DIR).
# Run with --help for usage. Rerunning resumes an interrupted download and skips finished parts.
set -euo pipefail

# The Replica scenes that have OpenLex3D labels (vMAP names them room_0, ..., OpenLex3D room0, ...)
SCENES=(room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4)
SEQUENCES=(00 01)  # 00 is the NICE-SLAM camera path, 01 a second path through the same scene
N_FRAMES=2000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
DL_DIR="$DATA_DIR/downloads"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

usage() {
    cat <<EOF
Usage: scripts/download_data.sh

Downloads, for the 8 Replica scenes in OpenLex3D:
  OpenLex3D labels   names and text queries per object       20 MB  -> \$DATA_DIR/gt_openlex3d/
  vMAP Replica       per frame: RGB, depth, pose and          45 GB  -> \$DATA_DIR/vmap/
                     instance mask, 2 x 2000 frames per scene

Needs about 95 GB free while downloading and 46 GB afterwards.

Environment:
  DATA_DIR=path     where to put the data (default: $REPO_ROOT/data)
  KEEP_ARCHIVES=1   keep the downloaded zips after extraction (default: delete them)
EOF
}

fetch() {  # fetch URL DIR: download into DIR, resuming a partial file
    wget --continue --quiet --show-progress --progress=bar:force:noscroll --directory-prefix "$2" "$1"
}

cleanup() {
    [[ $KEEP_ARCHIVES == 1 ]] || rm -f "$@"
}

missing=0
need() {
    [[ -e $1 ]] || { echo "  missing: $1" >&2; missing=1; }
}

need_frames() {  # need_frames DIR PATTERN: DIR must hold N_FRAMES files matching PATTERN
    local n
    n=$(find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l)
    ((n == N_FRAMES)) || { echo "  $1: expected $N_FRAMES files matching $2, found $n" >&2; missing=1; }
}

finish() {  # finish NAME OUT_DIR: fail if a check failed, otherwise mark OUT_DIR as complete
    if ((missing)); then
        echo "$1: extraction incomplete, see missing files above" >&2
        exit 1
    fi
    touch "$2/.complete"
    echo "$1: done -> $2"
}

get_labels() {
    local out="$DATA_DIR/gt_openlex3d"
    [[ -e $out/.complete ]] && { echo "labels: already done"; return; }
    echo "labels: downloading OpenLex3D labels"
    fetch http://aisdatasets.cs.uni-freiburg.de/openlex3d/gt_openlex3d.zip "$DL_DIR"
    unzip -qo "$DL_DIR/gt_openlex3d.zip" -d "$DATA_DIR"
    for s in "${SCENES[@]}"; do
        s="${s/_/}"  # room_0 -> room0
        need "$out/replica/$s/gt_categories.json"
        need "$out/replica/$s/gt_categories_query_to_object_mapping_all.json"
    done
    finish labels "$out"
    cleanup "$DL_DIR/gt_openlex3d.zip"
}

get_vmap() {
    local out="$DATA_DIR/vmap"
    [[ -e $out/.complete ]] && { echo "vmap: already done"; return; }
    echo "vmap: downloading Replica frames with instance masks"
    fetch https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip "$DL_DIR"
    echo "vmap: extracting (skipping the 10 GB of mesh textures, which only a renderer needs)"
    unzip -qo "$DL_DIR/vmap.zip" -x 'vmap/*/textures/*' -d "$DATA_DIR"
    for s in "${SCENES[@]}"; do
        need "$out/$s/habitat/info_semantic.json"
        need "$out/$s/habitat/mesh_semantic.ply"
        for q in "${SEQUENCES[@]}"; do
            local seq="$out/$s/imap/$q"
            need "$seq/traj_w_c.txt"
            need_frames "$seq/rgb" 'rgb_*.png'
            need_frames "$seq/depth" 'depth_*.png'
            need_frames "$seq/semantic_instance" 'semantic_instance_*.png'
        done
    done
    finish vmap "$out"
    cleanup "$DL_DIR/vmap.zip"
}

case "${1:-}" in
    "") ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
esac

mkdir -p "$DATA_DIR"
free_gb=$(($(df -Pk "$DATA_DIR" | awk 'NR == 2 {print $4}') / 1024 / 1024))
((free_gb >= 95)) || echo "warning: only $free_gb GB free in $DATA_DIR; the download needs about 95 GB" >&2

get_labels
get_vmap
rmdir "$DL_DIR" 2>/dev/null || true

cat <<EOF

Frames:  $DATA_DIR/vmap/<scene>/imap/{00,01}/, frame i = 0..1999 (not zero-padded, so sort by number)
         rgb/rgb_<i>.png, depth/depth_<i>.png (millimetres),
         semantic_instance/semantic_instance_<i>.png (Replica object ID per pixel),
         traj_w_c.txt (one camera-to-world 4x4 per line), 1200x680, fx = fy = 600, cx = 599.5, cy = 339.5
Names:   $DATA_DIR/vmap/<scene>/habitat/info_semantic.json (object ID -> Replica class)
Labels:  $DATA_DIR/gt_openlex3d/replica/<scene without underscore>/ (keyed by the same object IDs)
EOF
