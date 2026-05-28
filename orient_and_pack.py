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
GAP   = 8     # mm gap between parts — enough for build plate supports


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


def _overhang_score(mesh):
    """
    Total area of downward-facing faces that are not on the build plate.
    Uses PrusaSlicer's default 50° overhang threshold:
      a face needs support when its normal_z < -cos(50°) ≈ -0.643
    Lower score = less support material needed.
    """
    normals    = mesh.face_normals
    areas      = mesh.area_faces
    z_floor    = float(mesh.bounds[0][2])
    face_verts = mesh.vertices[mesh.faces]
    face_z_min = face_verts[:, :, 2].min(axis=1)

    # Faces pointing downward beyond 50° from horizontal, not touching the plate
    overhanging = (normals[:, 2] < -0.643) & (face_z_min > z_floor + 0.1)
    return float(areas[overhanging].sum())


def _sphere_fibonacci(n):
    """
    Generate n approximately uniformly distributed points on the unit sphere
    using the Fibonacci / golden-angle method.
    Each point represents a candidate 'down' direction to probe.
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i      = np.arange(n, dtype=float)
    theta  = 2.0 * np.pi * i / golden
    phi    = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def _cluster_normals(mesh, bin_size=0.05):
    """
    Group face normals by direction (binned to bin_size resolution) and
    return representative normals sorted by total coplanar area descending.
    Finds flat regions composed of many small triangles that no single face
    would reveal.
    """
    normals = mesh.face_normals
    areas   = mesh.area_faces
    bins    = {}
    for n, a in zip(normals, areas):
        key = tuple(np.round(n / bin_size).astype(int))
        if key not in bins:
            bins[key] = {'normal': n.copy(), 'area': 0.0}
        bins[key]['area'] += a
    return [c['normal'] for c in sorted(bins.values(), key=lambda c: c['area'], reverse=True)]


def _rotation_matrix_to_neg_z(src):
    """Return 3×3 rotation matrix that maps src vector → [0, 0, -1]."""
    src = np.asarray(src, dtype=float)
    src /= np.linalg.norm(src)
    dst = np.array([0.0, 0.0, -1.0])

    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-6:
        if np.dot(src, dst) > 0:
            return np.eye(3)
        # 180° flip around X
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

    axis  = cross / cross_norm
    angle = np.arccos(np.clip(np.dot(src, dst), -1.0, 1.0))
    c, s  = np.cos(angle), np.sin(angle)
    t     = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _score_candidate(normals, face_z_min, areas, R):
    """
    Compute overhang score for rotation R without copying or mutating the mesh.
    Rotates only the normals and face z-mins in numpy — fast for large meshes.
    """
    rot_normals  = normals @ R.T                    # (N, 3)
    rot_z_min    = face_z_min @ R[2]                # approximate: project face z-min onto new Z axis
    z_floor      = rot_z_min.min()
    overhanging  = (rot_normals[:, 2] < -0.643) & (rot_z_min > z_floor + 0.1)
    return float(areas[overhanging].sum())


def orient_largest_face_down(mesh):
    """
    Accurate support-minimising orientation search combining three strategies:

    1. Spherical Fibonacci (100 pts) — uniform global coverage so no good
       orientation is missed regardless of mesh topology.
    2. Coplanar-cluster candidates (top 20 by total flat-region area) — finds
       flat regions made of many small coplanar triangles correctly.
    3. Six OBB axis-aligned orientations — always tries the six obvious sides.

    Scoring uses the PrusaSlicer 50° overhang threshold without mesh.copy(),
    so it stays fast even on high-polygon parts.
    """
    normals    = mesh.face_normals.copy()       # (N, 3)
    areas      = mesh.area_faces.copy()         # (N,)
    face_verts = mesh.vertices[mesh.faces]      # (N, 3, 3)
    # Per-face minimum vertex position — used to detect faces on the build plate
    face_pts   = face_verts.reshape(-1, 3)      # (N*3, 3)  for fast projection

    # Build candidate pool
    sphere_pts = list(_sphere_fibonacci(100))
    clustered  = _cluster_normals(mesh)[:20]
    try:
        rot_mat  = mesh.bounding_box_oriented.primitive.transform[:3, :3]
        obb_axes = [rot_mat[:, i] * s for i in range(3) for s in (1, -1)]
    except Exception:
        obb_axes = []

    candidates = [np.asarray(c, dtype=float) for c in sphere_pts + clustered + obb_axes]

    # Precompute per-face min-vertex z-coordinate in a fast vectorisable form.
    # We approximate by keeping the raw vertex array and projecting per candidate.
    face_verts_flat = face_pts                  # (N*3, 3)

    best_score  = float('inf')
    best_normal = candidates[0].copy()

    for cand in candidates:
        R = _rotation_matrix_to_neg_z(cand)

        rot_normals = normals @ R.T                         # (N, 3)
        # Project all vertices onto new Z axis, reshape to (N, 3), take min per face
        proj_z      = face_verts_flat @ R[2]                # (N*3,)
        face_z_min  = proj_z.reshape(-1, 3).min(axis=1)    # (N,)
        z_floor     = face_z_min.min()

        overhanging = (rot_normals[:, 2] < -0.643) & (face_z_min > z_floor + 0.1)
        score = float(areas[overhanging].sum())

        if score < best_score:
            best_score  = score
            best_normal = cand.copy()

    _apply_rotation_to_align(mesh, best_normal, [0, 0, -1])
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


def try_rotate(mesh):
    """Rotate 90° in XY and return new footprint — does not mutate mesh."""
    w, d = footprint(mesh)
    return d, w  # swapped


def pack(parts):
    """
    Pack list of (name, mesh) onto as many plates as needed.
    Uses a shelf algorithm with a deferred-retry pass so parts that are too
    tall for the current shelf gap can be skipped and placed later if a
    shorter remaining space fits them.
    Returns list of plates; each plate is [(name, mesh, x_offset, y_offset)].
    """
    # Largest footprint first
    remaining = sorted(parts, key=lambda p: footprint(p[1])[0] * footprint(p[1])[1], reverse=True)

    plates = []

    while remaining:
        cur_plate  = []
        shelf_x    = 0.0
        shelf_y    = 0.0
        shelf_h    = 0.0
        unplaced   = []

        for name, mesh in remaining:
            w, d = footprint(mesh)

            # Try XY rotation if it helps fit
            rw, rd = try_rotate(mesh)
            use_rotated = False
            if w > BED_X and rw <= BED_X:
                use_rotated = True
                w, d = rw, rd

            if w > BED_X or d > BED_Y:
                print(f"  WARNING: '{name}' ({w:.1f}x{d:.1f} mm) exceeds bed size — skipped")
                continue

            placed = False

            # Try current shelf position
            if shelf_x + w <= BED_X and shelf_y + d <= BED_Y:
                pass  # falls through to place below
            elif shelf_x + w > BED_X:
                # Try a new shelf
                new_shelf_y = shelf_y + shelf_h + GAP
                if new_shelf_y + d <= BED_Y:
                    shelf_y = new_shelf_y
                    shelf_x = 0.0
                    shelf_h = 0.0
                else:
                    unplaced.append((name, mesh))
                    continue
            else:
                unplaced.append((name, mesh))
                continue

            if use_rotated:
                mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
                # Re-centre XY after rotation
                b = mesh.bounds
                mesh.apply_translation([
                    -0.5 * (b[0][0] + b[1][0]),
                    -0.5 * (b[0][1] + b[1][1]),
                    0
                ])
                w, d = footprint(mesh)

            cur_plate.append((name, mesh, shelf_x, shelf_y))
            shelf_x += w + GAP
            shelf_h  = max(shelf_h, d)

        if cur_plate:
            plates.append(cur_plate)
        remaining = unplaced

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
