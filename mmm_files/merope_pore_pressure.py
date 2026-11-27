"""
Cubes of the runs for MMM: the pore classes of a run file, each defined by its
   -   radius
   -   porosity fraction
   -   pressure
with the confining pressure of the cube, thrown by Mérope in a periodic cube. The run files are
../../sciantix-official/utilities/HBFF/mmm_inputs/run_*.txt, written by `python3 pore_distribution.py`:
run_NFIR.txt is the reference, the NFIR disc before the anneal (log-normal fit of its measured pores); run_R*.txt are the
Latin hypercube (index.txt, data/mmm_design.txt of pore_distribution.py) in the two factors of the geometry, xi and s,
at the R_V and the temperature of the reference.

MMM solves every cube twice, on the same mesh:
    unit/   pressure 1 in every pore: s_hat_1, the factor of the geometry, also needed for the confinement P_h and to
            bring the run with the class pressures to zero stress (zero mean of the matrix, D5.2)
    pk/     the gas pressure p_k of every class of the run file: s_hat_B, with the load of the neighbours
main() writes the cubes of unit/ (bubbles, .geo, summary) and a copy of the run file next to them. The pores do not
depend on the pressure (Mérope places them from radius, porosity, seed and distMin), so set_pressures.py of
mm-opera-hpc/polydispersebubble/scripts writes pk/ from unit/ without Mérope.
The cubes already listed in cubes.txt of a run are kept: more seeds, or new runs (an extension of the hypercube), add
their cubes only. test_rve.py writes the cubes of the tests of the RVE (size of the cube, size of the mesh) with generate().

Layout
    OUT/unit/run_<id>/run_<id>.txt       copy of the run file (class pressures, read by set_pressures.py)
    OUT/unit/run_<id>/unit_<id>.txt      the same classes at pressure 1, the input of generate()
    OUT/unit/run_<id>/cubes.txt          cubes of the run
    OUT/unit/run_<id>/side<L>_seed<NN>/  bubbles_unit_<id>.txt   one row per pore: phase, x, y, z, radius, pressure
                                         mesh_unit_<id>.geo      geometry for gmsh, mesh refined next to the pores (refine())
                                         summary_unit_<id>.txt   cube side, confining pressure, porosity and pores per class

Usage (from this folder, with Env_Merope.sh loaded)
    python3 merope_pore_pressure.py                     every run of the hypercube (index.txt but NFIR), SEEDS cubes each
    python3 merope_pore_pressure.py R05 R13 --seeds 3   these runs only, cubes of the seeds 1 to 3

Then bash run_gmsh.sh OUT, python3 inverted_elements.py OUT (ROUTINE.md, points b and b bis), and python3 set_pressures.py OUT/unit for pk/.

@ author: E.Cappellari, POLIMI-CEA 2026
@ date:   October 2026
"""

# %% ------------------------------------------------------------------------------------------------------------------
# Libraries and paths
import argparse
import filecmp
import glob
import itertools
import os
import re
import shutil
import sys
import sac_de_billes as sdb # Sphere generator
import merope
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_FILES = os.path.join(HERE, "..", "../sciantix-official/utilities/HBFF/mmm_inputs", "run_*.txt")  # run files of pore_distribution.py
INDEX = os.path.join(os.path.dirname(RUN_FILES), "index.txt")  # the runs of the hypercube, with their factors

# %% ------------------------------------------------------------------------------------------------------------------
# Model choices: RVE is dimensionless, nevertheless dimensions of objects within the RVE should stay consistent through 
# each other: lengths are in um, as the radii of the run file. The side of the cube, the repulsion distance and the 
# mesh size are fixed.
dimensions_side = 30  # um - side of the cube
distMin = 0.3         # um - repulsion distance, one HBS grain, from OperaHPC D5.2  
mesh_size = 0.15      # um - mesh size next to the pores or in the whole cube, as OperaHPC and the size test
SEEDS = 1             # cubes per run (seeds 1 to SEEDS), random positioning of the pores; the runs with few large pores
                      # may need more (one cube holds the FGR within 5 % in every run, not the mean of the largest class)
UNIT_PRESSURE = 1.0   # pressure of every pore of the cubes of unit/ (s_hat = sigma_I / p)

# %% ------------------------------------------------------------------------------------------------------------------
# Model choices: Mesh refinement next to the pores only, mesh_size is kept up to D_FINE from the pore surfaces, then 
# the size grows over D_TRANS up to FAR_TO_FINE * mesh_size (refine()).
REFINE = True
FAR_TO_FINE = 3.0     # size far from the pores / mesh_size (0.15 -> 0.45 um)
D_FINE = 0.3          # um - fine layer around each pore, as distMin: every ligament between close pores is fine
D_TRANS = 0.6         # um - growth from the fine size to the far one
R_FINE = 0.45         # um - next to a pore smaller than R_FINE the size is mesh_size * R / R_FINE (pore_size()), so that
                      # every pore has R / h >= R_FINE / mesh_size. 0 = off

# %% ------------------------------------------------------------------------------------------------------------------
# Folder name
MESH_RULE = ("_graded" + (f"_rfine{R_FINE:g}" if R_FINE > 0 else "")) if REFINE else "_uniform"
OUT = f"lhs_side{dimensions_side:g}_mesh{mesh_size:g}{MESH_RULE}"  # in the current directory

# %% ------------------------------------------------------------------------------------------------------------------
# Functions

###### Set of all triples between radius, pore fraction, pressure ######
def read_run(path):
    """Classes of a run file:
    - radius (um),
    - pore fraction (-),
    - pressure (MPa),
    - confining pressure (MPa)."""
    try:
        data = np.atleast_2d(np.loadtxt(path))
    except ValueError:
        raise SystemExit(f"{path}: not a run file (rows of 4 numbers, comments starting with #)")
    if data.shape[1] != 4:
        raise SystemExit(f"{path}: 4 columns expected (radius, porosity, pressure, confinement), found {data.shape[1]}")
    if np.ptp(data[:, 3]) > 0.0:
        raise SystemExit(f"{path}: the confining pressure differs between the rows")
    data = data[np.argsort(-data[:, 0])]
    return data[:, 0], data[:, 1], data[:, 2], float(data[0, 3])


def write_unit_run(path, radius, pore_fraction, title):
    """A run file in the format of read_run() with UNIT_PRESSURE in every class and no confinement."""
    lines = [f"# {title}", f"# pressure {UNIT_PRESSURE:g} in every class, no confinement",
             f"# {'radius_um':>10s}{'porosity':>14s}{'pressure_MPa':>14s}{'confinement_MPa':>17s}"]
    lines += [f"{r:12.6f}{pf:14.6e}{UNIT_PRESSURE:14.4f}{0.0:17.4f}" for r, pf in zip(radius, pore_fraction)]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def pores_in_cube(radius, pore_fraction, side):
    """Radius of the mean pore volume and expected number of pores per class in a cube of the given side. A class of porosity pf and pore volume V has pf / V pores per unit volume."""
    volume = 4.0 / 3.0 * np.pi * radius**3
    density = pore_fraction / volume  # pores per um3, per class
    r_v = np.cbrt(3.0 * pore_fraction.sum() / density.sum() / (4.0 * np.pi))
    return r_v, density * side**3


def pore_size(radius, h_fine=None, r_fine=None):
    """Mesh size next to pores of the given radius (um): h_fine (default mesh_size), times radius / r_fine below r_fine
    (default R_FINE; h_fine for every pore if r_fine <= 0)."""
    h_fine = mesh_size if h_fine is None else h_fine
    r_fine = R_FINE if r_fine is None else r_fine
    radius = np.asarray(radius, dtype=float)
    return h_fine * np.minimum(1.0, radius / r_fine) if r_fine > 0.0 else np.full_like(radius, h_fine)


def mesh_text(mesh, r_min):
    """Mesh of a cube as the summary writes it after "mesh size <mesh> um": nothing for the uniform size of Mérope; with
    REFINE the size far from the pores and, if the smallest radius r_min is below R_FINE, the size scaled with the radius
    (refine()). plot_bubbles.py and mesh_compare.py read "next to the pores, <far> um after" and "x R / <R_FINE> um",
    check_rule() compares the whole text to keep one rule in a folder."""
    if not REFINE:
        return ""
    h_min = float(pore_size(r_min, mesh))
    scaled = (f"; {mesh:g} um x R / {R_FINE:g} um next to the pores below R = {R_FINE:g} um ({h_min:.3f} um at "
              f"R = {r_min:.3f} um)" if h_min < mesh else "")
    return f" next to the pores, {FAR_TO_FINE * mesh:.3f} um after {D_FINE + D_TRANS:g} um from them{scaled}"


def rule_note():
    """The rule of the mesh, for the header of cubes.txt."""
    if not REFINE:
        return ""
    return (f"; mesh_um is the size next to the pores, {FAR_TO_FINE:g} times larger from {D_FINE + D_TRANS:g} um from "
            "them" + (f", mesh_um x R / {R_FINE:g} um next to the pores smaller than R = {R_FINE:g} um"
                      if R_FINE > 0 else "") + " (refine() of merope_pore_pressure.py)")


def refine(geo, centers, radii, side, h_fine=None, h_far=None, d_fine=D_FINE, d_trans=D_TRANS, r_fine=None):
    """Mesh refined next to the pores only, written in the .geo of Mérope (in place).

    Mérope gives one size to the whole cube (MeshSize {:}). Here every pore, with its periodic images that reach the
    cube [0, side]^3, gets a Gmsh Ball field: size h = pore_size(R, h_fine, r_fine) up to d_fine from the pore surface
    (h_fine, default mesh_size, or less next to the pores smaller than r_fine), then a linear growth over d_trans up to
    h_far (default FAR_TO_FINE * h_fine). The size is the smallest of the balls.
    Dropped from the .geo: MeshSize {:} (not used), Mesh 2; Mesh 3; (run_gmsh.sh meshes with -3: with them the volume
    was meshed twice) and the fields of a previous refine().
    centers (N x 3) and radii (N) in um. For a cube already written:
        b = np.loadtxt("bubbles_<run>.txt"); refine("mesh_<run>.geo", b[:, 1:4], b[:, 4], side)

    """
    h_fine = mesh_size if h_fine is None else h_fine
    h_far = FAR_TO_FINE * h_fine if h_far is None else h_far
    r_fine = R_FINE if r_fine is None else r_fine
    radii = np.asarray(radii, dtype=float)
    sizes = pore_size(radii, h_fine, r_fine)
    lines = []
    n = 0
    for c0, r, h in zip(np.asarray(centers, dtype=float), radii, sizes):
        for shift in itertools.product((-side, 0.0, side), repeat=3):
            c = c0 + shift
            gap = np.linalg.norm(np.maximum(0.0, np.maximum(-c, c - side)))  # from the centre to the cube
            if gap > r + d_fine + d_trans:
                continue
            n += 1
            lines += [f"Field[{n}] = Ball;",
                      f"Field[{n}].XCenter = {c[0]:.12g}; Field[{n}].YCenter = {c[1]:.12g}; Field[{n}].ZCenter = {c[2]:.12g};",
                      f"Field[{n}].Radius = {r + d_fine:.12g}; Field[{n}].Thickness = {d_trans:.12g};",
                      f"Field[{n}].VIn = {h:.12g}; Field[{n}].VOut = {h_far:.12g};"]
    lines += [f"Field[{n + 1}] = Min;",
              f"Field[{n + 1}].FieldsList = {{1:{n}}};",
              f"Background Field = {n + 1};",
              "Mesh.MeshSizeFromPoints = 0;",
              "Mesh.MeshSizeFromCurvature = 0;",
              "Mesh.MeshSizeExtendFromBoundary = 0;",
              f"Mesh.MeshSizeMax = {h_far:.12g};"]

    with open(geo) as f:
        text = f.read()
    text = re.sub(r"\n*// BEGIN  Refinement next to the pores.*?// END  Refinement next to the pores\n?", "\n", text,
                  flags=re.DOTALL)
    text = re.sub(r"^\s*MeshSize\s*\{:\}\s*=.*;\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Mesh\s+[23]\s*;\s*$", "", text, flags=re.MULTILINE)
    small = sizes < h_fine
    scaled = (f"; below R = {r_fine:g} um {h_fine:g} um x R / {r_fine:g} um ({sizes.min():.3g} um at R = "
              f"{radii.min():.3g} um, {np.count_nonzero(small)} pores)" if small.any() else "")
    header = [("// BEGIN  Refinement next to the pores (refine() of merope_pore_pressure.py): "
               f"{h_fine:g} um up to {d_fine:g} um from the pores, {h_far:g} um after {d_trans:g} um more{scaled}")]
    with open(geo, "w") as f:
        f.write(text.rstrip() + "\n\n" + "\n".join(header + lines) + "\n// END  Refinement next to the pores\n")
    print(f"Mesh refined next to the pores: {h_fine:g} um up to {d_fine:g} um from the pores, {h_far:g} um after "
          f"{d_trans:g} um more{scaled} ({n} balls with the periodic images)")
    return n


def generate(path, side=dimensions_side, seed=1, mesh=mesh_size, folder=None):
    """Pores of a run file in a periodic cube of the given side (um), placed with the given seed and meshed at the given
    size (um; with REFINE the size next to the pores, see refine()), and the files of the cube, written in folder (by
    default mmm_inputs/<run>_seed_<seed>_dmin<distMin> of the current directory)."""
    run = os.path.splitext(os.path.basename(path))[0]
    radius, pore_fraction, pressure, confinement = read_run(path)
    r_v, expected = pores_in_cube(radius, pore_fraction, side)
    L = [side, side, side]  # cube

    refined = mesh_text(mesh, radius.min())
    h_min = float(pore_size(radius.min(), mesh)) if REFINE else mesh  # size next to the smallest pore
    print(f"\n{run}: porosity {pore_fraction.sum():.4f}, R_V = {r_v:.4f} um, cube side {side:g} um with {expected.sum():.0f} pores, "
          f"distMin {distMin:.3f} um, mesh size {mesh:.3f} um{refined}, confining pressure {confinement:g} MPa")
    print(f"{'radius_um':>10s}{'porosity':>12s}{'pressure_MPa':>14s}{'pores expected':>16s}")
    for r, pf, p, n in zip(radius, pore_fraction, pressure, expected):
        print(f"{r:10.4f}{pf:12.4e}{p:14.2f}{n:16.1f}")
    if h_min > 0.5 * radius.min():
        print(f"mesh size {h_min:.3f} um next to the smallest pore, above half of its radius ({radius.min():.3f} um): the "
              "smallest class is poorly resolved")

    ###### Folder for the case: run, seed, dmin ######
    parent_folder = "mmm_inputs/"
    folder_name = folder or parent_folder + f"{run}_seed_{seed}_dmin{distMin:.3f}"
    os.makedirs(folder_name, exist_ok=True)

    ###### Generation of spheres for each triple between radius, pore fraction, pressure ######
    radius_pore_fraction = [[float(r), float(pf)] for r, pf in zip(radius, pore_fraction)]
    theSpheres = sdb.throwSpheres_3D(
        sdb.TypeAlgo.RSA,     # RSA algorithm will try to fullfil the porosity required by the user (upper limit),
                              # RSAmax instead will reach the theoretical limit of maximum porosity (approx 40%)
        sdb.NameShape.Tore,   # Tore - cubic shape for covering surface pores
        L,
        seed,
        radius_pore_fraction,
        [2] * len(radius),    # TBC, phase name, 1 = solid matrix, 2 = pores
        distMin
        )

    # the class of a sphere is the one with the nearest radius: its pressure follows
    classes = [int(np.argmin(np.abs(radius - sphere.radius))) for sphere in theSpheres]

    # pores are not meshed by including their id into a list
    tab_phase_not_to_mesh = []
    for j in range(0, len(theSpheres)):
        sphere = theSpheres[j]  # sphere is a structure containing the sphere properties
        sphere.phase = 2 + j    # id for each sphere
        tab_phase_not_to_mesh.append(sphere.phase)

    # Write sphere information to bubbles_<run>.txt in the dedicated folder
    with open(os.path.join(folder_name, f"bubbles_{run}.txt"), 'w') as bbl:
        for sphere, k in zip(theSpheres, classes):
            bbl.write(str(sphere.phase) + " ")
            bbl.write(str(sphere.center[0]) + " " +
                      str(sphere.center[1]) + " " +
                      str(sphere.center[2]) + " ")
            bbl.write(str(sphere.radius) + " ")
            bbl.write(str(pressure[k]) + "\n")

    # Porosity obtained per class, against the one asked (RSA stays below it), and the load of the cube
    placed = np.bincount(classes, minlength=len(radius))
    obtained = placed * 4.0 / 3.0 * np.pi * radius**3 / side**3
    with open(os.path.join(folder_name, f"summary_{run}.txt"), 'w') as out:
        out.write(f"# {run}: seed {seed}, cube side {side:.6f} um, distMin {distMin:.6f} um, mesh size {mesh:.6f} um{refined}\n")
        out.write(f"# confining pressure {confinement:g} MPa\n")
        out.write(f"# pores {len(theSpheres)} (expected {expected.sum():.0f}), porosity {obtained.sum():.6f} (asked {pore_fraction.sum():.6f})\n")
        out.write(f"# {'radius_um':>10s}{'pressure_MPa':>14s}{'porosity_asked':>16s}{'porosity':>14s}{'pores_expected':>16s}{'pores':>8s}\n")
        for r, p, pf, got, n, m in zip(radius, pressure, pore_fraction, obtained, expected, placed):
            out.write(f"{r:12.6f}{p:14.4f}{pf:16.6e}{got:14.6e}{n:16.1f}{m:8d}\n")
    print(f"{len(theSpheres)} pores placed, porosity {obtained.sum():.4f} (asked {pore_fraction.sum():.4f})")

    # Mesh generation setup (do not change)
    sphInc = merope.SphereInclusions_3D()
    sphInc.setLength(L)
    sphInc.setSpheres(theSpheres)
    mi = merope.MultiInclusions_3D()
    mi.setInclusions(sphInc)
    mi.setMatrixPhase(1)
    meshGenerator = merope.mesh.MeshGenerator()
    meshGenerator.setMeshOrder(2)  # perfect spheres
    meshGenerator.setMeshSize(mesh)  # mesh size Z
    meshGenerator.setMultiInclusions(mi)
    meshGenerator.do_not_mesh(tab_phase_not_to_mesh)

    # Set output file names
    # meshGenerator.set_nameOutput([os.path.join(folder_name, "mesh_" + run + ".vtk")])
    geo = os.path.join(folder_name, "mesh_" + run + ".geo")
    meshGenerator.write(geo)
    if REFINE:
        refine(geo, [[s.center[0], s.center[1], s.center[2]] for s in theSpheres], [s.radius for s in theSpheres],
               side, h_fine=mesh)

    print(f"Geometry generation completed: {folder_name}")


###### Cubes of the runs: cubes.txt of a run folder ######
def listed_cubes(out):
    """Rows of cubes.txt of a run folder: dicts with test, side, seed, mesh, folder; [] if there is no cubes.txt."""
    path = os.path.join(out, "cubes.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = [line.split() for line in f if line.strip() and not line.startswith("#")]
    return [{"test": t, "side": float(L), "seed": int(n), "mesh": float(h), "folder": f} for t, L, n, h, f in rows]


def check_rule(out, rows, r_min):
    """Exit if a cube of out was meshed with another rule than the settings at the top of this file (REFINE,
    FAR_TO_FINE, D_FINE, D_TRANS, R_FINE): the cubes of a folder must follow one rule."""
    for row in rows:
        summaries = glob.glob(os.path.join(out, row["folder"], "summary_*.txt"))
        if not summaries:
            continue
        with open(summaries[0]) as f:
            head = f.readline()
        expected = f"mesh size {row['mesh']:.6f} um{mesh_text(row['mesh'], r_min)}"
        if expected not in head:
            found = head.split("mesh size", 1)[-1].strip()
            raise SystemExit(f"{row['folder']} of {out} was meshed with another rule:\n  found    mesh size {found}\n"
                             f"  settings {expected}\nwrite the cubes in another folder (OUT) or delete these ones")


def runs_of_index(path=INDEX):
    """Ids of the runs listed in index.txt of pore_distribution.py (NFIR first, then the hypercube)."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: python3 pore_distribution.py writes it")
    ids = []
    with open(path) as f:
        for line in f:
            words = line.split()
            if words and not line.lstrip().startswith("#") and words[-1] == f"run_{words[0]}.txt":
                ids.append(words[0])
    return ids


def write_cubes(run, seeds=SEEDS, out_root=OUT):
    """Cubes of the seeds 1 to seeds of run at unit pressure, in out_root/unit/run_<run>, those not in cubes.txt yet."""
    path = os.path.normpath(os.path.join(os.path.dirname(RUN_FILES), f"run_{run}.txt"))
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: python3 pore_distribution.py writes it")
    radius, pore_fraction = read_run(path)[:2]
    out = os.path.abspath(os.path.join(out_root, "unit", f"run_{run}"))
    os.makedirs(out, exist_ok=True)
    rows = listed_cubes(out)
    check_rule(out, rows, radius.min())
    written = {row["folder"] for row in rows}

    # the run file next to the cubes: if it changed, the cubes were made for another design
    copy = os.path.join(out, f"run_{run}.txt")
    if os.path.exists(copy) and not filecmp.cmp(path, copy, shallow=False):
        raise SystemExit(f"{path} differs from its copy {copy}: the run changed after its cubes were written "
                         "(design of pore_distribution.py written again?); write the cubes in another OUT")
    shutil.copyfile(path, copy)
    unit = os.path.join(out, f"unit_{run}.txt")
    write_unit_run(unit, radius, pore_fraction, f"classes of run_{run}.txt at unit pressure, for the LHS "
                   "(merope_pore_pressure.py)")

    side = dimensions_side
    expected = pores_in_cube(radius, pore_fraction, side)[1]
    print(f"run_{run}: {expected.sum():.0f} pores expected in the cube of {side:g} um, {expected.min():.2f} in the "
          "rarest class")
    if expected.min() < 2.0:
        print(f"Mérope places the integer part of the expected number of pores of a class: the rarest class of run_{run} "
              "is missing or has one pore in a cube")
    for seed in range(1, seeds + 1):
        folder = f"side{side:g}_seed{seed:02d}"
        if folder in written:
            print(f"{folder}: already in {out}, kept (delete its folder and its row in cubes.txt to write it again)")
            continue
        generate(unit, side=side, seed=seed, mesh=mesh_size, folder=os.path.join(out, folder))
        rows.append({"test": "lhs", "side": side, "seed": seed, "mesh": mesh_size, "folder": folder})

    lines = [(f"# cubes of run_{run} of the Latin hypercube (python3 merope_pore_pressure.py), at unit pressure; "
              f"set_pressures.py writes the same cubes with the class pressures of run_{run}.txt in pk/{rule_note()}"),
             f"# {'test':>6s}{'side_um':>11s}{'seed':>6s}{'mesh_um':>9s}  folder"]
    lines += [f"{row['test']:>8s}{row['side']:11.6f}{row['seed']:6d}{row['mesh']:9g}  {row['folder']}"
              for row in sorted(rows, key=lambda r: r["folder"])]
    with open(os.path.join(out, "cubes.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(rows)} cubes of run_{run} listed in {os.path.join(out, 'cubes.txt')}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cubes of the runs of the Latin hypercube at unit pressure (unit/)")
    parser.add_argument("runs", nargs="*", help="ids of the runs (default: every run of index.txt but NFIR)")
    parser.add_argument("--seeds", type=int, default=SEEDS, help=f"cubes per run, seeds 1 to N (default {SEEDS})")
    args = parser.parse_args(argv)
    listed = runs_of_index()
    runs = args.runs or [r for r in listed if r != "NFIR"]
    unknown = [r for r in runs if r not in listed]
    if unknown:
        raise SystemExit(f"runs not in {INDEX}: {', '.join(unknown)}")
    print(f"{len(runs)} runs, {args.seeds} cube(s) each, in {os.path.abspath(OUT)}/unit")
    for run in runs:
        write_cubes(run, args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
