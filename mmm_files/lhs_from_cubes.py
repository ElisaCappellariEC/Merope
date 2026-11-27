"""
Check of the cubes of a tree of Mérope against the design of the MMM runs. Reads the copy run_<id>.txt that
merope_pore_pressure.py leaves in every run folder of FOLDER/unit and checks that
    - the runs of the tree are those of the design (data/mmm_design.txt of pore_distribution.py),
    - the classes of every copy are those that pore_distribution.py writes for the run: radius and porosity (the
      geometry of the cubes) and load (what set_pressures.py takes for the pk cubes),
    - the runs are a Latin hypercube in (xi, s): one run per stratum of each factor in the box MMM_FACTORS,
and writes FOLDER/lhs_design.txt, one row per run: xi, s, strata, checks.
A load that differs is reported, not an error: the copies older than 27/11/2026 have other loads (gas closure a = 2
until 24/11, the f of the third factor eta_bar until 27/11); set_pressures.py needs the current ones, c_dp / R: copy
mmm_inputs/run_<id>.txt into the run folder first.
Why (27/11/2026): the random pairing of the appended runs R20-R34 of the old pore_distribution.py was not the same on
two PCs; the cubes and meshes were made with the one of the other PC, now fixed in data/mmm_design.txt.

Usage (from this folder, Python of pore_distribution.py)
    python3 lhs_from_cubes.py [FOLDER]        default lhs_side30_mesh0.15_graded_rfine0.45

@ author: E.Cappellari, POLIMI-CEA 2026
@ date:   November 2026
"""

# %% ------------------------------------------------------------------------------------------------------------------
# Libraries and paths
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sciantix-official", "utilities", "HBFF"))
import pore_distribution  # noqa: E402  design of the runs and their classes

# %% ------------------------------------------------------------------------------------------------------------------
# Settings
FOLDER = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "lhs_side30_mesh0.15_graded_rfine0.45")
TABLE = os.path.join(FOLDER, "lhs_design.txt")

# %% ------------------------------------------------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    # %% the design and the copies of the tree --------------------------------------------------------------------------
    cases = pore_distribution.mmm_cases()
    lo, hi = np.array(list(pore_distribution.MMM_FACTORS.values())).T
    N = len(cases)
    unit = os.path.join(FOLDER, "unit")
    ids = sorted(name[len("run_"):] for name in os.listdir(unit) if name.startswith("run_R"))
    if ids != [c["id"] for c in cases]:
        raise SystemExit(f"runs in {unit}: {' '.join(ids)}; the design has {cases[0]['id']}-{cases[-1]['id']} "
                         f"({pore_distribution.MMM_DESIGN})")

    # %% every run: its strata and the checks of its classes -----------------------------------------------------------
    runs, problems, old_loads = [], [], []
    for c in cases:
        path = os.path.join(unit, f"run_{c['id']}", f"run_{c['id']}.txt")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found: merope_pore_pressure.py copies the run file next to the cubes")
        data = np.atleast_2d(np.loadtxt(path))
        reduced = pore_distribution.reduced_classes(c["classes"], c["T"], pore_distribution.MMM_CLASSES)
        classes = np.array([[q["R"] * 1e6, c["xi"] * q["xi_share"], pore_distribution.c_dp(c["T"]) / q["R"] / 1e6,
                             pore_distribution.MMM_PH / 1e6] for q in reduced])  # as mmm_run_files() writes them
        geometry = (data.shape == classes.shape and np.allclose(data[:, 0], classes[:, 0], rtol=0.0, atol=1e-6)
                    and np.allclose(data[:, 1], classes[:, 1], rtol=1e-5, atol=0.0))
        load = geometry and np.allclose(data[:, 2:], classes[:, 2:], rtol=1e-6, atol=1e-4)
        if not geometry:
            problems.append(f"{c['id']}: the radii or porosities of {path} are not those of the design")
        elif not load:
            old_loads.append(f"{c['id']}: the loads of {path} are not those of pore_distribution.py")
        x = np.array([c["xi"], c["classes"]["sigma"]])
        runs.append({"id": c["id"], "origin": c["origin"].split(",")[0], "x": x,
                     "strata": np.floor((x - lo) / (hi - lo) * N).astype(int), "geometry": geometry, "load": load})

    # %% the design: Latin hypercube in (xi, s) --------------------------------------------------------------------------
    strata = np.array([r["strata"] for r in runs])
    latin = all(np.array_equal(np.sort(strata[:, j]), np.arange(N)) for j in range(strata.shape[1]))
    if not latin:
        problems.append(f"the {N} runs are not a Latin hypercube of {N} levels in (xi, s) (two runs in one stratum)")

    # %% the table of the runs -----------------------------------------------------------------------------------------
    ranges = ", ".join(f"{name} {l:g}-{h:g}" for name, l, h in zip(pore_distribution.MMM_FACTORS, lo, hi))
    lines = [f"# Runs of {pore_distribution.MMM_DESIGN} against the cubes of {unit} (lhs_from_cubes.py, "
             f"{datetime.date.today()})",
             f"# {N} levels per factor in {ranges}: stratum k = floor((x - lo) / (hi - lo) {N})",
             "# geometry, load: the classes of the copy run_<id>.txt are those of pore_distribution.py for the run "
             "(radius and porosity; load)",
             f"# {'run':5s}{'origin':>10s}{'xi':>9s}{'s':>9s}{'k_xi':>6s}{'k_s':>5s}{'geometry':>10s}{'load':>6s}"]
    for r in runs:
        lines.append(f"  {r['id']:5s}{r['origin']:>10s}{r['x'][0]:9.4f}{r['x'][1]:9.4f}"
                     + "".join(f"{k:{w}d}" for k, w in zip(r["strata"], (6, 5)))
                     + f"{'ok' if r['geometry'] else 'NO':>10s}{'ok' if r['load'] else 'NO':>6s}")
    lines.append(f"# Latin hypercube of {N} levels in (xi, s): {'yes' if latin else 'NO'}")
    with open(TABLE, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written {TABLE}")
    for line in old_loads:
        print(f"note: {line} (copy mmm_inputs/run_<id>.txt into the run folder before set_pressures.py)")
    if problems:
        print("\n".join(f"PROBLEM: {line}" for line in problems))
        raise SystemExit(f"{len(problems)} problems")
