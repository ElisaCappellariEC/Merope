"""
Cubes with pores of one radius and no close neighbour, to check the stress factor of MMM against known values:
- isolated: two pores 8.7 um apart in a cube of 10 um (porosity 0.2 %, interaction (R / d)^3 / 2 = 2e-4): the hoop stress at the wall is p / 2 (Lamé);
- regular:  the simple cubic array of porosity XI, equal pores at the same pressure: the "regular arrangement" of the Biot stress of OperaHPC D5.2 (footnote 2: s_hat = 1/2 + xi); with the mean stress of the matrix xi / (1 - xi) p amplified by 3/2 at a spherical wall, 1/2 + 3/2 xi / (1 - xi).

The regular array is written as a cell of 2 x 2 x 2 pores, the same periodic problem as one pore in a cell of side a, whose 8 values also give the noise of the mesh. The mesh follows the rule of the runs (refine() of merope_pore_pressure.py: mesh_size next to the pores, graded after).
MMM must solve them with the options of the runs (run_mmm.sh: -sf 0 -dl 0.15).

Usage (from this folder, with Env_Merope.sh loaded)
    python3 test_single_pore.py                          writes the cubes not written yet (bubbles, .geo)
    bash run_gmsh.sh OUT && bash ~/mmm/build/polydispersebubble/run_mmm.sh OUT
    python3 test_single_pore.py                          again: prints s_hat of the cubes solved in ~/mmm/results/OUT

@ author: E.Cappellari, POLIMI-CEA 2026
@ date:   November 2026
"""

# %% ------------------------------------------------------------------------------------------------------------------
# Libraries and paths
import itertools
import os
import sys
import merope
import merope_pore_pressure as mpp
import numpy as np
import sac_de_billes as sdb

R_PORE = 0.61            # um: radius of the pores
XI = 0.1131              # porosity of the cubes of the size test (run_NFIR)
SIDE_ISOLATED = 10.0     # um: side of the cube of the isolated pores
A = R_PORE * np.cbrt(4.0 * np.pi / (3.0 * XI))  # um: side of the cell of one pore of the regular array
CUBES = {"isolated": (SIDE_ISOLATED, [[0.25 * SIDE_ISOLATED] * 3, [0.75 * SIDE_ISOLATED] * 3]),
         "regular": (2.0 * A, [[A * (0.5 + i), A * (0.5 + j), A * (0.5 + k)]
                               for i, j, k in itertools.product((0, 1), repeat=3)])}  # side (um), centres of the pores
NAME = "single"          # bubbles_<NAME>.txt, mesh_<NAME>.geo
OUT = f"test_single_pore_mesh{mpp.mesh_size:g}{mpp.MESH_RULE}"  # in the current directory, solved in ~/mmm/results/OUT
RESULTS = os.path.join(os.path.expanduser("~"), "mmm", "results", OUT)

# %% ------------------------------------------------------------------------------------------------------------------
# Main
def main():
    # %% 1. cubes not written yet ----------------------------------------------------------------------------------------
    for case, (side, centres) in CUBES.items():
        folder = os.path.join(OUT, case)
        if os.path.isfile(os.path.join(folder, f"mesh_{NAME}.geo")):
            print(f"{folder}: already written, kept")
            continue
        os.makedirs(folder, exist_ok=True)
        spheres = [sdb.Sphere_3D(c, R_PORE, 2 + j) for j, c in enumerate(centres)]  # phase = id, as generate()
        with open(os.path.join(folder, f"bubbles_{NAME}.txt"), "w") as f:
            for j, c in enumerate(centres):
                f.write(f"{2 + j} {c[0]:.12g} {c[1]:.12g} {c[2]:.12g} {R_PORE:.12g} {mpp.UNIT_PRESSURE:g}\n")
        xi = len(centres) * 4.0 / 3.0 * np.pi * R_PORE**3 / side**3
        with open(os.path.join(folder, f"summary_{NAME}.txt"), "w") as f:
            f.write(f"# {case}: {len(centres)} pores R = {R_PORE:g} um, cube side {side:.6f} um, porosity {xi:.6f}, "
                    f"mesh size {mpp.mesh_size:g} um{mpp.mesh_text(mpp.mesh_size, R_PORE)}\n")
        inclusions = merope.SphereInclusions_3D()
        inclusions.setLength([side] * 3)
        inclusions.setSpheres(spheres)
        multi = merope.MultiInclusions_3D()
        multi.setInclusions(inclusions)
        multi.setMatrixPhase(1)
        generator = merope.mesh.MeshGenerator()
        generator.setMeshOrder(2)
        generator.setMeshSize(mpp.mesh_size)
        generator.setMultiInclusions(multi)
        generator.do_not_mesh([s.phase for s in spheres])
        geo = os.path.join(folder, f"mesh_{NAME}.geo")
        generator.write(geo)
        if mpp.REFINE:
            mpp.refine(geo, centres, [R_PORE] * len(centres), side, h_fine=mpp.mesh_size)
        print(f"{folder}: {len(centres)} pores, cube side {side:.4f} um, porosity {xi:.4f}")

    # %% 2. s_hat of the cubes solved --------------------------------------------------------------------------------------
    # s_hat = (sigma + h) / (1 - h), sigma = sigma_I_max / p at zero strain, h = xi - (1 - xi) <sigma_h> / p (plot_bubbles.py)
    for case, (side, centres) in CUBES.items():
        folder = os.path.join(RESULTS, case)
        stresses, hydro = (os.path.join(folder, f) for f in ("bubbles_and_stresses_selected.txt", "sig_hydro.txt"))
        if not (os.path.isfile(stresses) and os.path.isfile(hydro)):
            print(f"{folder}: not solved yet")
            continue
        sigma = np.atleast_1d(np.genfromtxt(stresses, delimiter=",", names=True)["sigma_I_max"]) / mpp.UNIT_PRESSURE
        volume, average = (np.genfromtxt(hydro, delimiter=",", names=True)[k].item() for k in ("Volume", "Average"))
        xi = 1.0 - volume / side**3
        h = xi - (1.0 - xi) * average / mpp.UNIT_PRESSURE
        s_hat = (sigma + h) / (1.0 - h)
        print(f"{case:9s} xi {xi:.4f}  h {h:.4f}  sigma {sigma.mean():.4f} ({sigma.min():.4f}-{sigma.max():.4f})  "
              f"s_hat {s_hat.mean():.4f} ({s_hat.min():.4f}-{s_hat.max():.4f})  | isolated 0.5, 1/2 + xi {0.5 + xi:.4f}, "
              f"1/2 + 3/2 xi/(1-xi) {0.5 + 1.5 * xi / (1.0 - xi):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
