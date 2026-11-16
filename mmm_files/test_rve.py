"""
Cubes for two tests of the RVE of the runs for MMM, written by generate() of merope_pore_pressure.py
- the size of the cube, on the reference run NFIR of ../mmm_inputs: the NFIR disc before the anneal, the log-normal fit of
  its measured pores (not the SCIANTIX state);
- the size of the mesh, on the cube of seed 1 with the sizes of MESH_SIZES: the first is that of the size test, the
  others in the constant ratio sqrt(2) (two coarser, one finer), so that plot_bubbles.py measures the error of the mesh
  of the runs against the finest and extrapolates with Richardson on the three finest when the class means allow it.
This script writes the geometry only (bubbles, .geo, summary) with Mérope. It writes the cubes not listed yet in
cubes.txt of the folder: a size added to MESH_SIZES, or more SEEDS, add their cubes only.

Both tests follow Kanit et al., Int. J. Solids Struct. 40 (2003) 3647: the size of the cube is Sections 5.2 and 6 (Eq. 62, 68-70), the size of the mesh is Section 4.1.2.
Usage
    python3 test_rve.py

@ author: E.Cappellari, POLIMI-CEA 2026
@ date:   October 2026
"""

# %% ------------------------------------------------------------------------------------------------------------------
# Libraries and paths
import os
import sys
import merope_pore_pressure
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# %% ------------------------------------------------------------------------------------------------------------------
# Model choices
RUN = "NFIR"                      # run of ../mmm_inputs: the reference, the NFIR disc before the anneal (pore_distribution.py)
SIDE = 30.0                       # um: side of the cubes
MESH_SIZE = 0.15                  # um: mesh size of the cubes, next to the pores with REFINE of merope_pore_pressure.py
SEEDS = 10                        # cubes of the size test, as the ten seeds of OperaHPC D5.2
EPS_R = 0.05                      # relative precision asked on the mean of s_hat of a class
MESH_SIZES = (MESH_SIZE, MESH_SIZE * 2**0.5, 2 * MESH_SIZE, MESH_SIZE / 2**0.5)  # um: meshes of the mesh test on the
                                  # cube of seed 1; MESH_SIZES[0] is the mesh of the size test, a new size goes at the END

# Result of the mesh test (16/11/2026, test_rve_side30_mesh0.15_graded_rfine0.45_delta0.15, -dl 0.15; README 9.4):
# s_hat decreases with refinement in every class. From 0.15 to 0.106 um the classes of the verdict move by <= 1.15 %
# and x_c of the FGR by <= 1 %; against h -> 0 the 0.15 mesh overestimates s_hat by ~2-3.5 % (Richardson works on the
# classes 0.610-1.192, orders 1.0-1.9; not on 0.436, not yet asymptotic, nor on the classes of 11 pores, noise). The
# error is nearly the same in every class, so it goes into sigma_R: 0.15 um is kept for the LHS. The 0.106 mesh has
# 2.6 x the vertices of 0.15 (6.4 M): partition-mesh ~39 GB, MMM with 2 cores per process.
EPS_MESH = EPS_R                 # tolerance on s_hat between the mesh of the runs and the finest mesh
mpp = merope_pore_pressure
PRESSURE = mpp.UNIT_PRESSURE      # unit pressure of every pore (s_hat = sigma_I / p), no distribution

# folder of the cubes, in the current directory; its name holds the rule of the mesh of merope_pore_pressure.py, so that
# cubes meshed with different rules are not mixed (write_cubes() also checks the summaries of the cubes already there)
OUT = f"test_rve_side{SIDE:g}_mesh{MESH_SIZE:g}{mpp.MESH_RULE}"

# %% ------------------------------------------------------------------------------------------------------------------
# Functions
def run_file(mpp, run):
    """The run file of the run: where merope_pore_pressure.py takes the runs (RUN_FILES), or ../mmm_inputs of this folder."""
    for folder in (os.path.dirname(mpp.RUN_FILES), os.path.join(HERE, "..", "mmm_inputs")):
        path = os.path.normpath(os.path.join(folder, f"run_{run}.txt"))
        if os.path.exists(path):
            return path
    raise SystemExit(f"run_{run}.txt is not in the folder of RUN_FILES of merope_pore_pressure.py nor in ../mmm_inputs: "
                     "python3 ../pore_distribution.py")


def mesh_variant(h):
    """Folder suffix of the cube of seed 1 with the mesh size h: meshdouble for 2 MESH_SIZE, mesh<h> otherwise."""
    return "meshdouble" if np.isclose(h, 2 * MESH_SIZE) else f"mesh{h:.3g}"


def write_cubes(mpp, run):
    """Cubes of the run with mpp, the module merope_pore_pressure, that cubes.txt does not list yet."""
    path = run_file(mpp, run)
    radius, pore_fraction = mpp.read_run(path)[:2]
    out = os.path.abspath(os.path.join(OUT, f"run_{run}"))
    os.makedirs(out, exist_ok=True)
    rows = mpp.listed_cubes(out)  # the meshes of the mesh test, and the cubes of the size test, follow one rule
    mpp.check_rule(out, rows, radius.min())
    written = {row["folder"] for row in rows}

    # size of the cube: the classes of the run at unit pressure, in the cube of SIDE; the mesh test: the cube of seed 1
    # with the other sizes of MESH_SIZES
    side = SIDE
    unit = os.path.join(out, f"unit_{run}.txt")
    mpp.write_unit_run(unit, radius, pore_fraction,
                       f"classes of {os.path.basename(path)} for the size test of the RVE (test_rve.py)")
    expected = mpp.pores_in_cube(radius, pore_fraction, side)[1]
    print(f"run_{run} ({path}): {expected.sum():.0f} pores expected in the cube of {side:g} um, {expected.min():.2f} in the rarest class")
    if expected.min() < 2.0:
        print(f"Mérope places the integer part of the expected number of pores of a class: the rarest class of run_{run} "
              "is missing or has one pore in a cube")
    cubes = [("size", seed, MESH_SIZES[0], f"side{side:g}_seed{seed:02d}") for seed in range(1, SEEDS + 1)]
    cubes += [("mesh", 1, h, f"side{side:g}_seed01_{mesh_variant(h)}") for h in MESH_SIZES[1:]]
    for test, seed, h, folder in cubes:
        if folder in written:
            print(f"{folder}: already in {out}, kept (delete its folder and its row in cubes.txt to write it again)")
            continue
        mpp.generate(unit, side=side, seed=seed, mesh=h, folder=os.path.join(out, folder))
        rows.append({"test": test, "side": side, "seed": seed, "mesh": h, "folder": folder})

    lines = [(f"# cubes of run_{run} for the tests of the RVE (python3 test_rve.py): MMM solves every folder "
              f"with the pressure {PRESSURE:g} in all the pores, plot_bubbles.py reads the results{mpp.rule_note()}"),
             f"# {'test':>6s}{'side_um':>11s}{'seed':>6s}{'mesh_um':>9s}  folder"]
    lines += [f"{row['test']:>8s}{row['side']:11.6f}{row['seed']:6d}{row['mesh']:9g}  {row['folder']}" for row in rows]
    with open(os.path.join(out, "cubes.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(rows)} cubes of run_{run} listed in {os.path.join(out, 'cubes.txt')}")


def main():
    """The cubes not written yet."""
    write_cubes(merope_pore_pressure, RUN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
