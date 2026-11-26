"""
Placement map of the runs appended to the Latin hypercube of pore_distribution.py: for every pair (xi, s) of the
levels that mmm_appended() can give to an appended run, whether Mérope places its pores with the call of generate() of
merope_pore_pressure.py (RSA, periodic cube of side dimensions_side, distMin), with the seeds SEEDS.
In the dense runs the RSA of Mérope stops ("The grid has been too much subdivided (more than NB_MAX_SUB_VOXELS
voxels)") when the excluded volume sum_k xi_k ((R_k + distMin / 2) / R_k)^3 goes above the jamming limit of RSA, which
is lower for the narrow widths: the appended runs of 25/11/2026 R20, R22, R23 (xi 0.272, 0.248, 0.296 with s 0.108,
0.066, 0.235) were not placed. mmm_appended() pairs the levels only on the pairs placed with every seed.
Geometry only: eta_bar changes the load, not radii and porosities, so every pair is made at ETA.

Output: MMM_PLACEMENT of pore_distribution.py (sciantix-official/utilities/HBFF/data/merope_placement.txt), one row
per pair: xi, s, excluded volume, placed with each seed of SEEDS (1 or 0), porosity lost with the first seed (%, nan if
not placed). Write it again when the design of pore_distribution.py (MMM_RUNS, MMM_SEED, MMM_FACTORS, MMM_APPENDED,
MMM_XI_MAX, MMM_CLASSES) or dimensions_side, distMin of merope_pore_pressure.py change: mmm_appended() stops if the
levels of the map are not its own.

Usage (from this folder, with Env_Merope.sh loaded)
    python3 placement_map.py        ≈ 10 min with PROCESSES = 4

@ author: E.Cappellari, POLIMI-CEA 2026
@ date:   November 2026
"""

# %% ------------------------------------------------------------------------------------------------------------------
# Libraries and paths
import datetime
import multiprocessing
import os
import sys

import numpy as np
import sac_de_billes as sdb  # Sphere generator
import merope_pore_pressure as mpp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sciantix-official", "utilities", "HBFF"))
import pore_distribution  # noqa: E402  design of the runs, MMM_PLACEMENT

# %% ------------------------------------------------------------------------------------------------------------------
# Settings
SEEDS = (1, 2)  # seeds of the cubes tried: mmm_appended() takes the pairs placed with all of them
ETA = 0.15      # eta_bar of the pairs, it does not change the geometry
PROCESSES = 4   # pairs placed at the same time: a stopped RSA takes 2-3 GB (15 at once were killed on 16 GB)

# %% ------------------------------------------------------------------------------------------------------------------
# Functions

def placement(task):
    """Row of the map for task = (xi, s, R_V, T): xi, s, excluded volume, placed with each seed (1/0), porosity lost
    with the first seed (%)."""
    xi, s, R_V, T = task
    c = pore_distribution.mmm_case("map", "placement map", xi, s, ETA, 0.0, R_V, T)
    classes = pore_distribution.reduced_classes(c["classes"], T, pore_distribution.MMM_CLASSES)
    # radius and porosity of the classes as the run file writes them, largest class first as read_run()
    radius = np.array([float(f"{k['R'] * 1e6:.6f}") for k in classes])
    pore_fraction = np.array([float(f"{xi * k['xi_share']:.6e}") for k in classes])
    order = np.argsort(-radius)
    radius, pore_fraction = radius[order], pore_fraction[order]
    excluded = np.sum(pore_fraction * ((radius + mpp.distMin / 2.0) / radius) ** 3)
    side = mpp.dimensions_side
    placed, lost = [], np.nan
    sys.stdout.flush()
    saved, null = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)  # Mérope writes its log on the standard output
    try:
        for seed in SEEDS:
            try:
                spheres = sdb.throwSpheres_3D(sdb.TypeAlgo.RSA, sdb.NameShape.Tore, [side, side, side], seed,
                                              [[float(r), float(pf)] for r, pf in zip(radius, pore_fraction)],
                                              [2] * len(radius), mpp.distMin)
            except RuntimeError:
                placed.append(0)
                continue
            placed.append(1)
            if seed == SEEDS[0]:
                obtained = sum(4.0 / 3.0 * np.pi * sphere.radius**3 for sphere in spheres) / side**3
                lost = 100.0 * (1.0 - obtained / pore_fraction.sum())
    finally:
        os.dup2(saved, 1)
        os.close(null)
        os.close(saved)
    return (xi, s, excluded, *placed, lost)

# %% ------------------------------------------------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    cases = pore_distribution.mmm_design(pore_distribution.MMM_RUNS)
    box, _, empty = pore_distribution.mmm_appended_levels(cases)
    lo, hi = np.array(list(box.values())).T
    xi_levels, s_levels = lo[0] + empty[0] * (hi[0] - lo[0]), lo[1] + empty[1] * (hi[1] - lo[1])
    R_V, T = cases[0]["R_V"], cases[0]["T"]
    tasks = [(xi, s, R_V, T) for xi in xi_levels[::-1] for s in s_levels]  # the densest first, the slowest
    print(f"{len(tasks)} pairs (xi {xi_levels.min():.3f}-{xi_levels.max():.3f}, s {s_levels.min():.3f}-"
          f"{s_levels.max():.3f}), seeds {SEEDS}, {PROCESSES} at a time")
    with multiprocessing.Pool(PROCESSES) as pool:
        rows = sorted(pool.map(placement, tasks, chunksize=1))

    seeds = "  ".join(f"seed{seed}" for seed in SEEDS)
    lines = [f"# Placement by Merope of the runs appended to the Latin hypercube (Merope/mmm_files/placement_map.py, "
             f"{datetime.date.today()})",
             f"# RSA in a periodic cube of {mpp.dimensions_side:g} um, distMin {mpp.distMin:g} um, seeds "
             f"{' '.join(str(seed) for seed in SEEDS)}; design of pore_distribution.py: MMM_RUNS {pore_distribution.MMM_RUNS}, "
             f"MMM_SEED {pore_distribution.MMM_SEED}, MMM_APPENDED {pore_distribution.MMM_APPENDED}, MMM_XI_MAX "
             f"{pore_distribution.MMM_XI_MAX:g}, {pore_distribution.MMM_CLASSES} classes, R_V {R_V * 1e6:.4g} um",
             "# placed: 1 = throwSpheres_3D returned the pores, 0 = it stopped (above the jamming limit of RSA); "
             "excluded volume = sum_k xi_k ((R_k + distMin / 2) / R_k)^3",
             f"# {'xi':>8s}{'s':>10s}{'excluded':>10s}  {seeds}  porosity_lost_%"]
    lines += [f"{xi:10.6f}{s:10.6f}{excluded:10.4f}" + "".join(f"{p:7d}" for p in placed) + f"{lost:10.3f}"
              for xi, s, excluded, *placed, lost in rows]
    with open(pore_distribution.MMM_PLACEMENT, "w") as f:
        f.write("\n".join(lines) + "\n")

    every = {(xi, s) for xi, s, _, *placed, _ in rows if all(placed)}
    print(f"{len(every)} of {len(rows)} pairs placed with every seed; narrowest s placed per xi:")
    for xi in xi_levels:
        placed_s = [s for s in s_levels if (xi, s) in every]
        print(f"  xi {xi:.3f}: s >= {min(placed_s):.3f}" if placed_s else f"  xi {xi:.3f}: none")
    print(f"written {pore_distribution.MMM_PLACEMENT}")
