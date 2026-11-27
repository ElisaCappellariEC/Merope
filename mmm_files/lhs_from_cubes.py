"""
The design of the Latin hypercube as the cubes of Mérope were made. Reads the copy run_<id>.txt that
merope_pore_pressure.py leaves in every run folder of FOLDER/unit, finds the levels (xi, s, eta_bar) of every run and
checks that
    - the copies of R00-R19 are the runs of the hypercube of pore_distribution.py (mmm_design()),
    - the appended runs (R20 on) sit on the empty strata of mmm_appended_levels(), one per stratum,
    - the classes of every copy are those of mmm_case() for its levels (radius, porosity; load),
    - the runs together are a Latin hypercube of MMM_RUNS + MMM_APPENDED levels,
    - Mérope places every appended pair (xi, s) with every seed of MMM_PLACEMENT (placement_map.py),
and writes
    FOLDER/lhs_design.txt         one row per run: factors, strata, checks, distance in (xi, s) to the nearest run
    HBFF/data/mmm_appended.txt    strata of the appended runs, only when all the checks of the geometry pass
Why: mmm_appended() orders the xi levels with np.argsort of their counts of placed pairs, which has ties; the order of
the ties depends on numpy, so pore_distribution.py gave two pairings on two PCs (27/11/2026) and the cubes were made
with the one of the other PC. mmm_appended.txt keeps the pairing of the cubes (pore_distribution.py does not read it
yet). eta_bar only sets f, the scale of the loads: it does not change the geometry nor s_hat.

A level of an appended run comes from the header of its copy (xi with 6 digits, s and eta_bar with 4) and is
confirmed by its classes, to the digits of the file. A load of R00-R19 that differs is reported, not an error: the
copies of 10/11/2026 have the loads of the gas closure a = 2 (set_pressures.py needs the current ones).

Usage (from this folder, Python of pore_distribution.py)
    python3 lhs_from_cubes.py [FOLDER]        default lhs_side30_mesh0.15_graded_rfine0.45

@ author: E.Cappellari, POLIMI-CEA 2026
@ date:   November 2026
"""

# %% ------------------------------------------------------------------------------------------------------------------
# Libraries and paths
import datetime
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sciantix-official", "utilities", "HBFF"))
import pore_distribution  # noqa: E402  design of the runs, MMM_PLACEMENT

# %% ------------------------------------------------------------------------------------------------------------------
# Settings
FOLDER = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "lhs_side30_mesh0.15_graded_rfine0.45")
TABLE = os.path.join(FOLDER, "lhs_design.txt")
APPENDED = os.path.join(pore_distribution.DATA_DIR, "mmm_appended.txt")
HEADER = re.compile(r"# xi = (\S+), s = (\S+) \(CV_n = [^)]*\), eta_bar = ([^,]+)")

# %% ------------------------------------------------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    # %% the design of pore_distribution.py: the runs of the hypercube and the empty strata of the appended runs -------
    cases = pore_distribution.mmm_design(pore_distribution.MMM_RUNS)
    box, u_old, empty = pore_distribution.mmm_appended_levels(cases)
    lo, hi = np.array(list(box.values())).T
    n_old, n_new = len(cases), pore_distribution.MMM_APPENDED
    N = n_old + n_new
    R_V, T = cases[0]["R_V"], cases[0]["T"]
    placement = np.atleast_2d(np.loadtxt(pore_distribution.MMM_PLACEMENT))

    unit = os.path.join(FOLDER, "unit")
    copies = sorted(glob.glob(os.path.join(unit, "run_R*", "run_R*.txt")))
    ids = [os.path.basename(p)[len("run_"):-len(".txt")] for p in copies]
    expected = [f"R{k:0{len(str(N - 1))}d}" for k in range(N)]
    if ids != expected:
        raise SystemExit(f"runs in {unit}: {' '.join(ids)}; expected {expected[0]}-{expected[-1]}")

    # %% every run: its levels, its strata and the checks of its classes ---------------------------------------------
    runs, problems, old_loads = [], [], []
    for k, (run, path) in enumerate(zip(ids, copies)):
        with open(path) as fh:
            text = fh.read().splitlines()
        heads = [m for m in (HEADER.match(line) for line in text) if m]
        if not heads:
            raise SystemExit(f"{path}: no line '# xi = ..., s = ... (CV_n = ...), eta_bar = ...'")
        data = np.atleast_2d(np.loadtxt(path, comments="#"))
        if k < n_old:  # a run of the hypercube: the levels of mmm_design()
            c, u = cases[k], u_old[k]
            origin = "hypercube"
        else:  # an appended run: the strata of the values of its header, which must be empty strata
            head = heads[0].groups()
            u = (np.floor((np.array([float(v) for v in head]) - lo) / (hi - lo) * N) + 0.5) / N
            level = lo + u * (hi - lo)  # as mmm_appended(): lo + u (hi - lo), u = (stratum + 0.5) / N
            for j, name in enumerate(box):
                if not np.any(u[j] == empty[j]):
                    problems.append(f"{run}: {name} {head[j]} in a stratum of the hypercube, not an empty one")
            if (f"{level[0]:g}", f"{level[1]:.4g}", f"{level[2]:.4g}") != head:
                problems.append(f"{run}: header xi, s, eta_bar {', '.join(head)}, the centres of their strata "
                                f"{level[0]:g}, {level[1]:.4g}, {level[2]:.4g}")
            c = pore_distribution.mmm_case(run, "copy of the cubes", R_V=R_V, T=T, Ph_rel=pore_distribution.MMM_PH_REL,
                                           **dict(zip(box, level)))
            origin = "appended"
        reduced = pore_distribution.reduced_classes(c["classes"], T, pore_distribution.MMM_CLASSES)
        classes = np.array([[q["R"] * 1e6, c["xi"] * q["xi_share"], c["f"] * pore_distribution.c_dp(T) / q["R"] / 1e6,
                             c["Ph"] / 1e6] for q in reduced])  # as mmm_run_files() writes them
        geometry = (data.shape == classes.shape and np.allclose(data[:, 0], classes[:, 0], rtol=0.0, atol=1e-6)
                    and np.allclose(data[:, 1], classes[:, 1], rtol=1e-5, atol=0.0))
        load = geometry and np.allclose(data[:, 2:], classes[:, 2:], rtol=1e-6, atol=1e-4)
        if not geometry:
            problems.append(f"{run}: the radii or porosities of {path} are not those of its levels")
        elif not load:
            (old_loads if origin == "hypercube" else problems).append(f"{run}: the loads of {path} are not those of "
                                                                      "pore_distribution.py")
        placed = ""
        if origin == "appended":
            row = np.flatnonzero((np.abs(placement[:, 0] - c["xi"]) < 1e-5)
                                 & (np.abs(placement[:, 1] - c["classes"]["sigma"]) < 1e-5))
            placed = "yes" if len(row) == 1 and np.all(placement[row[0], 3:-1] == 1) else "no"
            if placed == "no":
                problems.append(f"{run}: (xi, s) = ({c['xi']:g}, {c['classes']['sigma']:.4g}) not placed with every "
                                f"seed of {pore_distribution.MMM_PLACEMENT}")
        runs.append({"id": run, "origin": origin, "xi": c["xi"], "s": c["classes"]["sigma"], "eta": c["eta"],
                     "f": c["f"], "u": u, "strata": np.floor(u * N).astype(int), "geometry": geometry, "load": load,
                     "placed": placed})

    # %% the design: Latin hypercube, distances in (xi, s) ----------------------------------------------------------
    u_all = np.array([r["u"] for r in runs])
    latin = all(np.array_equal(np.sort(np.floor(u_all[:, j] * N).astype(int)), np.arange(N)) for j in range(len(box)))
    if not latin:
        problems.append(f"the {N} runs are not a Latin hypercube of {N} levels (two runs in one stratum)")
    d = np.linalg.norm(u_all[:, None, :2] - u_all[None, :, :2], axis=-1)  # (xi, s) in the unit box
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    a = n_old + int(np.argmin(nearest[n_old:]))
    b = int(np.argmin(d[a]))
    gap = nearest[a]  # as gap() of mmm_appended(): an appended run against every other run

    # %% the table of the runs -----------------------------------------------------------------------------------------
    ranges = ", ".join(f"{name} {l:g}-{h:g}" for name, l, h in zip(("xi", "s", "eta_bar"), lo, hi))
    lines = [f"# Runs of the Latin hypercube as the cubes of {unit} were made (lhs_from_cubes.py, {datetime.date.today()})",
             f"# {N} levels per factor in {ranges}: stratum k, level = lo + (k + 0.5) / {N} (hi - lo)",
             "# geometry, load: the classes of the copy run_<id>.txt are those of mmm_case() for the levels (radius and "
             "porosity; load); placed: by Merope with every seed (MMM_PLACEMENT)",
             "# nearest: distance in (xi, s), unit box, to the nearest other run",
             f"# {'run':5s}{'origin':>10s}{'xi':>9s}{'s':>9s}{'eta_bar':>9s}{'f':>8s}{'k_xi':>6s}{'k_s':>5s}{'k_eta':>6s}"
             f"{'geometry':>10s}{'load':>6s}{'placed':>8s}{'nearest':>9s}"]
    for r, near in zip(runs, nearest):
        lines.append(f"  {r['id']:5s}{r['origin']:>10s}{r['xi']:9.4f}{r['s']:9.4f}{r['eta']:9.4f}{r['f']:8.3f}"
                     + "".join(f"{k:{w}d}" for k, w in zip(r["strata"], (6, 5, 6)))
                     + f"{'ok' if r['geometry'] else 'NO':>10s}{'ok' if r['load'] else 'NO':>6s}{r['placed'] or '-':>8s}"
                       f"{near:9.4f}")
    lines += [f"# Latin hypercube of {N} levels: {'yes' if latin else 'NO'}; smallest distance of an appended run "
              f"{gap:.4f} ({runs[a]['id']}-{runs[b]['id']}); smallest among the runs of the hypercube "
              f"{nearest[:n_old].min() if n_old else np.nan:.4f}"]
    with open(TABLE, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written {TABLE}")

    # %% the strata of the appended runs, when the geometry of every run is the one of its levels -------------------
    for line in old_loads:
        print(f"note: {line} (copy mmm_inputs/run_<id>.txt into the run folder before set_pressures.py)")
    if problems:
        print("\n".join(f"PROBLEM: {line}" for line in problems))
        raise SystemExit(f"{len(problems)} problems: {APPENDED} not written")
    lines = [f"# Strata of the runs appended to the Latin hypercube of pore_distribution.py, as the cubes of {unit} "
             f"were made (lhs_from_cubes.py, {datetime.date.today()})",
             f"# level = lo + (k + 0.5) / {N} (hi - lo) in {ranges}; MMM_RUNS {n_old}, MMM_SEED "
             f"{pore_distribution.MMM_SEED}, MMM_APPENDED {n_new}, MMM_XI_MAX {pore_distribution.MMM_XI_MAX:g}",
             f"# all placed by Merope with every seed (MMM_PLACEMENT); smallest distance in (xi, s) of an appended run "
             f"{gap:.4f} ({runs[a]['id']}-{runs[b]['id']})",
             f"# {'run':5s}{'k_xi':>6s}{'k_s':>5s}{'k_eta':>6s}{'xi':>10s}{'s':>10s}{'eta_bar':>10s}"]
    lines += [f"  {r['id']:5s}" + "".join(f"{k:{w}d}" for k, w in zip(r["strata"], (6, 5, 6)))
              + f"{r['xi']:10.6f}{r['s']:10.6f}{r['eta']:10.6f}" for r in runs[n_old:]]
    with open(APPENDED, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"written {APPENDED}")
