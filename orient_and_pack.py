"""
Orient all STLs for least support material on an MK4S (250x210mm bed),
then pack as many parts as fit per plate and export each plate as one STL.

Geometry is never modified — only rotation and translation.
"""

import trimesh
import numpy as np
from pathlib import Path

BED_X = 245   # usable width  (250mm - 5mm margin)
BED_Y = 205   # usable depth  (210mm - 5mm margin)
GAP   = 3     # mm gap between parts


# ── Orientation helpers ───────────────────────────────────────────────────────

def _apply_rotation_to_align(mesh, src_vec, dst_vec):
    """Rotate mesh so src_vec aligns with dst_vec."""
    src = np.array(src_vec, dtype=float)
    dst = np.array(dst_vec, dtype=float)
    src /= np.linalg.norm(src)
    dst /= np.linalg.norm(dst)

    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-6:
        if np.dot(src, dst) > 0:
            return  # already aligned
        # 180-degree flip — pick any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(src[0]) < 0.9 else np.array([0, 1, 0])
        R = trimesh.transformations.rotation_matrix(np.pi, perp)
    else:
        axis  = cross / cross_norm
        angle = np.arccos(np.clip(np.dot(src, dst), -1.0, 1.0))
        R     = trimesh.transformations.rotation_matrix(angle, axis)

    mesh.apply_transform(R)


def orient_vertical(mesh):
    """Stand the part on its shortest end (longest axis = Z). Best for augers/screws."""
    obb       = mesh.bounding_box_oriented
    extents   = obb.extents
    rot_mat   = obb.primitive.transform[:3, :3]
    long_idx  = int(np.argmax(extents))
    long_axis = rot_mat[:, long_idx]

    _apply_rotation_to_align(mesh, long_axis, [0, 0, 1])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])


def orient_largest_face_down(mesh):
    """Place the largest flat face on the build plate — minimises overhangs."""
    idx    = int(np.argmax(mesh.area_faces))
    normal = mesh.face_normals[idx].copy()

    # We want that face's outward normal pointing DOWN (-Z)
    _apply_rotation_to_align(mesh, normal, [0, 0, -1])
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])


VERTICAL_KEYWORDS = ('screw', 'auger')   # cylindrical / threaded parts — stand upright


def orient(mesh, name):
    """Choose orientation strategy by part name, with fallback."""
    low = name.lower()
    try:
        if any(k in low for k in VERTICAL_KEYWORDS):
            orient_vertical(mesh)
        else:
            orient_largest_face_down(mesh)
    except Exception:
        # Fallback: align minimum bounding box face with build plate
        try:
            orient_largest_face_down(mesh)
        except Exception:
            # Last resort: just drop to Z=0 as-is
            mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    # Centre in XY so packing math is tidy
    b = mesh.bounds
    mesh.apply_translation([
        -0.5 * (b[0][0] + b[1][0]),
        -0.5 * (b[0][1] + b[1][1]),
        0
    ])


# ── Bin packing (shelf algorithm) ────────────────────────────────────────────

def footprint(mesh):
    b = mesh.bounds
    return (b[1][0] - b[0][0]), (b[1][1] - b[0][1])   # (width, depth)


def pack(parts):
    """
    Pack list of (name, mesh) onto as many plates as needed.
    Returns list of plates; each plate is [(name, mesh, x_offset, y_offset)].
    Tries both XY orientations of each part and picks whichever fits first.
    """
    # Largest footprint first
    parts_sorted = sorted(parts, key=lambda p: footprint(p[1])[0] * footprint(p[1])[1], reverse=True)

    plates     = []
    shelf_x    = 0.0
    shelf_y    = 0.0
    shelf_h    = 0.0
    cur_plate  = []

    def flush_plate():
        nonlocal shelf_x, shelf_y, shelf_h, cur_plate
        if cur_plate:
            plates.append(cur_plate)
        cur_plate = []
        shelf_x   = 0.0
        shelf_y   = 0.0
        shelf_h   = 0.0

    for name, mesh in parts_sorted:
        w, d = footprint(mesh)

        # If the part is wider than the bed, try rotating 90° in XY
        if w > BED_X and d <= BED_X:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
            w, d = footprint(mesh)

        if w > BED_X or d > BED_Y:
            print(f"  WARNING: '{name}' ({w:.1f}x{d:.1f} mm) exceeds bed size — skipped")
            continue

        # New shelf needed?
        if shelf_x + w > BED_X:
            shelf_y += shelf_h + GAP
            shelf_x  = 0.0
            shelf_h  = 0.0

        # New plate needed?
        if shelf_y + d > BED_Y:
            flush_plate()

        cur_plate.append((name, mesh, shelf_x, shelf_y))
        shelf_x += w + GAP
        shelf_h  = max(shelf_h, d)

    flush_plate()
    return plates


# ── Main ─────────────────────────────────────────────────────────────────────

root    = Path("/Users/rabiaakhtar/Downloads/cat")
out_dir = root / "plates"
out_dir.mkdir(exist_ok=True)

EXCLUDE = {'front_w_letters'} | {f'letter-{i}-' for i in range(1, 10)}

stl_files = sorted(
    p for p in root.rglob("*.stl")
    if "plates" not in str(p)
    and not any(p.stem == ex or p.stem.startswith(ex) for ex in EXCLUDE)
)
print(f"Found {len(stl_files)} STL files\n")

parts = []
for path in stl_files:
    name = path.stem
    print(f"  orienting: {name}")
    try:
        mesh = trimesh.load(str(path), force='mesh')
        orient(mesh, name)
        parts.append((name, mesh))
    except Exception as exc:
        print(f"    ERROR: {exc}")

print(f"\nPacking {len(parts)} parts onto {BED_X}x{BED_Y} mm bed (MK4S)…\n")
plates = pack(parts)
print(f"→ {len(plates)} plate(s) needed\n")

for i, plate in enumerate(plates, 1):
    meshes = []
    print(f"Plate {i} ({len(plate)} parts):")
    for name, mesh, x, y in plate:
        b   = mesh.bounds
        # offset so part sits at (x, y) from its min corner
        tx  = x - b[0][0]
        ty  = y - b[0][1]
        mesh.apply_translation([tx, ty, 0])
        meshes.append(mesh)
        w, d = footprint(mesh)
        print(f"    {name:40s}  {w:6.1f} x {d:6.1f} mm")

    combined = trimesh.util.concatenate(meshes)
    out_path = out_dir / f"plate_{i}.stl"
    combined.export(str(out_path))
    print(f"  → {out_path}\n")

print("Done. Drag the plate_*.stl files into PrusaSlicer.")
