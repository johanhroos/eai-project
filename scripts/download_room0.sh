#!/usr/bin/env bash
# Download only Replica room_0 for local OpenLex3D development.
#
# Downloads:
#   - OpenLex3D ground-truth labels
#   - vMAP archive
#   - Extracts ONLY room_0
#
# The large vMAP zip is deleted after extraction unless KEEP_ARCHIVES=1.

set -euo pipefail

SCENE="room_0"
SEQUENCES=(00 01)
N_FRAMES=2000

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
DL_DIR="$DATA_DIR/downloads"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

usage() {
    cat <<EOF
Usage: scripts/download_data_room0.sh

Downloads Replica room_0 for OpenLex3D.

Data:
  OpenLex3D labels
  vMAP room_0:
    - RGB
    - depth
    - camera poses
    - semantic instance masks
    - Habitat semantic information

Environment:
  DATA_DIR=path
      Dataset destination.
      Default: $REPO_ROOT/data

  KEEP_ARCHIVES=1
      Keep downloaded zip files.
      Default: delete after extraction.
EOF
}

fetch() {
    local url="$1"
    local dir="$2"

    mkdir -p "$dir"

    wget \
        --continue \
        --quiet \
        --show-progress \
        --progress=bar:force:noscroll \
        --directory-prefix "$dir" \
        "$url"
}

cleanup() {
    [[ "$KEEP_ARCHIVES" == "1" ]] || rm -f "$@"
}

missing=0

need() {
    [[ -e "$1" ]] || {
        echo "missing: $1" >&2
        missing=1
    }
}

need_frames() {
    local dir="$1"
    local pattern="$2"
    local n

    n=$(find "$dir" -maxdepth 1 -name "$pattern" 2>/dev/null | wc -l)

    if (( n != N_FRAMES )); then
        echo "$dir: expected $N_FRAMES files matching $pattern, found $n" >&2
        missing=1
    fi
}

get_labels() {

    local out="$DATA_DIR/gt_openlex3d"

    if [[ -e "$out/replica/room0/gt_categories.json" ]]; then
        echo "labels: room_0 labels already present"
        return
    fi

    echo "labels: downloading OpenLex3D labels"

    fetch \
        "http://aisdatasets.cs.uni-freiburg.de/openlex3d/gt_openlex3d.zip" \
        "$DL_DIR"

    echo "labels: extracting room0"

    unzip -qo \
        "$DL_DIR/gt_openlex3d.zip" \
        'gt_openlex3d/replica/room0/*' \
        -d "$DATA_DIR"

    need "$out/replica/room0/gt_categories.json"
    need "$out/replica/room0/gt_categories_query_to_object_mapping_all.json"

    if (( missing )); then
        echo "labels: extraction incomplete" >&2
        exit 1
    fi

    echo "labels: done"

    cleanup "$DL_DIR/gt_openlex3d.zip"
}

get_vmap() {

    local out="$DATA_DIR/vmap/$SCENE"

    if [[ -e "$out/.complete" ]]; then
        echo "vmap room_0: already done"
        return
    fi

    echo "vmap: downloading archive (~45 GB)"

    fetch \
        "https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip" \
        "$DL_DIR"

    echo "vmap: extracting ONLY room_0"

    # Extract everything belonging to room_0 except the large
    # renderer textures, which aren't needed by our pipeline.
    unzip -qo \
        "$DL_DIR/vmap.zip" \
        "vmap/$SCENE/*" \
        -x "vmap/$SCENE/textures/*" \
        -d "$DATA_DIR"

    echo "vmap: verifying room_0"

    need "$out/habitat/info_semantic.json"
    need "$out/habitat/mesh_semantic.ply"

    for q in "${SEQUENCES[@]}"; do

        local seq="$out/imap/$q"

        need "$seq/traj_w_c.txt"

        need_frames "$seq/rgb" \
            'rgb_*.png'

        need_frames "$seq/depth" \
            'depth_*.png'

        need_frames "$seq/semantic_instance" \
            'semantic_instance_*.png'

    done

    if (( missing )); then
        echo "vmap room_0: extraction incomplete" >&2
        exit 1
    fi

    touch "$out/.complete"

    echo "vmap room_0: done -> $out"

    echo "Removing large vMAP archive..."
    cleanup "$DL_DIR/vmap.zip"
}

case "${1:-}" in
    "")
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
esac

mkdir -p "$DATA_DIR"
mkdir -p "$DL_DIR"

get_labels
get_vmap

rmdir "$DL_DIR" 2>/dev/null || true

cat <<EOF

Done.

Dataset:

$DATA_DIR/
├── gt_openlex3d/
│   └── replica/
│       └── room0/
│
└── vmap/
    └── room_0/
        ├── habitat/
        │   ├── info_semantic.json
        │   └── mesh_semantic.ply
        │
        └── imap/
            ├── 00/
            │   ├── rgb/
            │   ├── depth/
            │   ├── semantic_instance/
            │   └── traj_w_c.txt
            │
            └── 01/
                ├── rgb/
                ├── depth/
                ├── semantic_instance/
                └── traj_w_c.txt

Camera:
    resolution = 1200 x 680
    fx = fy = 600
    cx = 599.5
    cy = 339.5

Depth is stored in millimetres.

EOF
