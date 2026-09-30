#!/usr/bin/env bash
# Extract Replica room_0 from an existing vMAP archive, without downloading anything.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
VMAP_ARCHIVE="$DATA_DIR/downloads/vmap.zip"
LABELS_ARCHIVE="$DATA_DIR/downloads/gt_openlex3d.zip"
SCENE_DIR="$DATA_DIR/vmap/room_0"
LABELS_DIR="$DATA_DIR/gt_openlex3d/replica/room0"

usage() {
    cat <<EOF
Usage: bash scripts/extract_room0.sh

Extracts only Replica room_0 from an existing vmap.zip into DATA_DIR/vmap/room_0.
Includes both camera sequences, RGB, depth, semantic masks, poses, and scene metadata.
Renderer textures are skipped. The archive is kept.

If the room0 OpenLex3D labels are missing, extracts them from
DATA_DIR/downloads/gt_openlex3d.zip (if present).

DATA_DIR defaults to $REPO_ROOT/data.
EOF
}

case "${1:-}" in
    "") ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
esac

if [[ ! -f "$VMAP_ARCHIVE" ]]; then
    echo "Missing vMAP archive: $VMAP_ARCHIVE" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "This script requires python3." >&2
    exit 1
fi

extract_prefix() {
    python3 - "$1" "$2" "$3" "${4:-}" <<'PY'
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

archive_path, destination, prefix, excluded_prefix = sys.argv[1:]
with ZipFile(archive_path) as archive:
    members = [
        info for info in archive.infolist()
        if info.filename.startswith(prefix)
        and (not excluded_prefix or not info.filename.startswith(excluded_prefix))
        and not info.is_dir()
    ]
    if not members:
        raise SystemExit(f"No files matching {prefix!r} in {archive_path}")
    for info in members:
        if ".." in PurePosixPath(info.filename).parts or info.filename.startswith("/"):
            raise SystemExit(f"Unsafe archive path: {info.filename}")
    for index, info in enumerate(members, 1):
        archive.extract(info, Path(destination))
        if index % 2000 == 0 or index == len(members):
            print(f"Extracted {index}/{len(members)} files", flush=True)
PY
}

require_file() {
    if [[ ! -s "$1" ]]; then
        echo "Missing or empty: $1" >&2
        exit 1
    fi
}

check_frames() {
    local dir="$1" prefix="$2" count
    if [[ ! -d "$dir" ]]; then
        echo "Missing directory: $dir" >&2
        exit 1
    fi
    count=$(find "$dir" -maxdepth 1 -type f -name "${prefix}_*.png" | wc -l)
    if (( count != 2000 )); then
        echo "Expected 2000 ${prefix} PNGs in $dir, found $count" >&2
        exit 1
    fi
    require_file "$dir/${prefix}_0.png"
    require_file "$dir/${prefix}_1999.png"
}

check_poses() {
    local file="$1"
    require_file "$file"
    if ! awk 'NF != 16 { invalid = 1 } END { if (NR != 2000 || invalid) exit 1 }' "$file"; then
        echo "Expected 2000 camera poses with 16 values each in $file" >&2
        exit 1
    fi
}

# The labels are already extracted in this project; use the archive only if needed.
if [[ ! -s "$LABELS_DIR/gt_categories.json" || ! -s "$LABELS_DIR/gt_categories_query_to_object_mapping_all.json" ]]; then
    if [[ ! -f "$LABELS_ARCHIVE" ]]; then
        echo "Missing room0 labels and label archive: $LABELS_ARCHIVE" >&2
        exit 1
    fi
    echo "Extracting OpenLex3D room0 labels..."
    extract_prefix "$LABELS_ARCHIVE" "$DATA_DIR" 'gt_openlex3d/replica/room0/'
fi
require_file "$LABELS_DIR/gt_categories.json"
require_file "$LABELS_DIR/gt_categories_query_to_object_mapping_all.json"

echo "Extracting Replica room_0 from $VMAP_ARCHIVE (about 6 GiB)..."
mkdir -p "$SCENE_DIR"
rm -f "$SCENE_DIR/.complete"
extract_prefix "$VMAP_ARCHIVE" "$DATA_DIR" 'vmap/room_0/' 'vmap/room_0/textures/'

require_file "$SCENE_DIR/habitat/info_semantic.json"
require_file "$SCENE_DIR/habitat/mesh_semantic.ply"
for seq in 00 01; do
    seq_dir="$SCENE_DIR/imap/$seq"
    check_poses "$seq_dir/traj_w_c.txt"
    check_frames "$seq_dir/rgb" rgb
    check_frames "$seq_dir/depth" depth
    check_frames "$seq_dir/semantic_instance" semantic_instance
done

touch "$SCENE_DIR/.complete"
echo "Verified room_0 data in $SCENE_DIR"
echo "Kept archive at $VMAP_ARCHIVE"
