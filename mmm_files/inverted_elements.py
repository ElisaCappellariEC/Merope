"""
Inverted elements (negative Jacobian) in the order-2 tetrahedra of gmsh meshes (MSH 2.2).
@ author: E.Cappellari, POLIMI-CEA 2026
@ date:   October 2026

Usage
    python3 inverted_elements.py FOLDER            every mesh_<name>.msh below FOLDER, with bubbles_<name>.txt next to it
    python3 inverted_elements.py MESH [BUBBLES]    one mesh
Exit 1 if a mesh has an inverted element.
"""
# %% ------------------------------------------------------------------------------------------------------------------
# Libraries and paths
import glob
import os
import sys

import numpy as np

EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]  # vertices of the nodes 4..9 of a gmsh 10-node tetrahedron
GRID = 10  # points i/GRID in barycentric coordinates: 286 points per element, vertices, edges and faces included

# %% ------------------------------------------------------------------------------------------------------------------
# Functions
def read_msh22(path):
    """Nodes (N x 3) and 10-node tetrahedra (E x 10, indices into the nodes) of a MSH 2.2 file."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    nodes = np.array([l.split() for l in lines[i + 2:i + 2 + n]], dtype=float)
    index = {int(k): j for j, k in enumerate(nodes[:, 0])}
    i = lines.index("$Elements")
    tets = []
    for l in lines[i + 2:i + 2 + int(lines[i + 1])]:
        p = l.split()
        if p[1] == "11":
            first = 3 + int(p[2])
            tets.append([index[int(v)] for v in p[first:first + 10]])
    return nodes[:, 1:4], np.array(tets, dtype=int).reshape(-1, 10)


def min_det_jacobian(xyz, tets):
    """Smallest det J of every element over the GRID points."""
    X = xyz[tets]
    dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    smallest = np.full(len(tets), np.inf)
    for i in range(GRID + 1):
        for j in range(GRID + 1 - i):
            for k in range(GRID + 1 - i - j):
                L = np.array([i, j, k, GRID - i - j - k]) / GRID
                dN = np.zeros((10, 3))
                for v in range(4):
                    dN[v] = (4 * L[v] - 1) * dL[v]
                for e, (a, b) in enumerate(EDGES):
                    dN[4 + e] = 4 * (L[b] * dL[a] + L[a] * dL[b])
                smallest = np.minimum(smallest, np.linalg.det(np.einsum("eai,aj->eij", X, dN)))
    return smallest


def check(mesh, bubbles=None):
    """Print the inverted elements of one mesh and the pore next to each; return their number."""
    xyz, tets = read_msh22(mesh)
    bad = np.where(min_det_jacobian(xyz, tets) < 0)[0]
    print(f"{mesh}: {len(tets)} tetrahedra, {len(bad)} inverted")
    if len(bad) == 0:
        return 0
    pores = np.loadtxt(bubbles, ndmin=2) if bubbles and os.path.isfile(bubbles) else None
    box = xyz.max(axis=0) - xyz.min(axis=0)  # periodic cube: distances to the nearest image of a pore
    for e in bad:
        centre = xyz[tets[e, :4]].mean(axis=0)
        if pores is None:
            print(f"   element at ({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}) um")
            continue
        d = centre - pores[:, 1:4]
        d -= box * np.round(d / box)
        gap = np.linalg.norm(d, axis=1) - pores[:, 4]
        k = np.argmin(gap)
        print(f"   element next to pore {int(pores[k, 0]):4d} (R {pores[k, 4]:.3f} um), "
              f"{gap[k]:+.3f} um from its surface")
    return len(bad)

# %% ------------------------------------------------------------------------------------------------------------------
# Main
def main(args):
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if os.path.isdir(args[0]):
        meshes = sorted(glob.glob(os.path.join(args[0], "**", "mesh_*.msh"), recursive=True))
        if not meshes:
            print(f"ERROR: no mesh_*.msh below {args[0]}")
            return 1
        found = {}
        for m in meshes:
            name = os.path.basename(m)[len("mesh_"):-len(".msh")]
            found[m] = check(m, os.path.join(os.path.dirname(m), f"bubbles_{name}.txt"))
        bad = [m for m, n in found.items() if n]
        print(f"\n{len(meshes)} meshes, {len(bad)} with inverted elements")
        for m in bad:
            print(f"   {found[m]:4d}  {m}")
        return 1 if bad else 0
    return 1 if check(args[0], args[1] if len(args) > 1 else None) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
