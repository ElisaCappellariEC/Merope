#!/bin/bash
# author: E.Cappellari, POLIMI-CEA 2026

# =============================================================================
# 3D mesh of every mesh_*.geo written by Mérope in the selected folder: with
# REFINE of merope_pore_pressure.py the .geo has the mesh refined next to the
# pores (refine()), otherwise the uniform size of Mérope.
#
# Outputs:
#   mesh_<name>.msh
#   mesh_<name>.log
#
# Usage, from the directory where the Python scripts were run
#   ./run_gmsh.sh              every mesh_*.geo below FOLDER (set below)
#   ./run_gmsh.sh OTHER_FOLDER every mesh_*.geo below OTHER_FOLDER only (e.g. one new cube)
# =============================================================================

set -uo pipefail

# =====CONFIGURATION============================================================
GMSH="/home/ecappellari/.local/bin/gmsh"
FOLDER="${1:-test_rve_side30_mesh0.15}"  # folder of the .geo files: the first argument, if given
MSH_FORMAT="2.2"
case "$MSH_FORMAT" in
    2.2) GMSH_FORMAT="msh2" ;;
    4.1) GMSH_FORMAT="msh41" ;;
    *)
        echo "ERROR: Unsupported MSH format version: $MSH_FORMAT" >&2
        exit 1
        ;;
esac
# High-order optimisation of the curved order-2 elements that get inverted (jac. < 0) near close bubbles:
# with them MMM stops with "solve: the non linear solver did not converge" or gives a wrong stress next to them.
#   2 = elastic + optimisation: needed at uniform 0.3 um (side10_seed01: 3 inverted elements fixed), but on the
#       refined mesh its global elastic system went over 14 GB of RAM (side10_seed01: killed by the system)
#   1 = optimisation of the bad patches only, no cost when there is none. Refined meshes, jac. < 0 before it:
#       side10_seed01 at 0.15 um none; side30_seed01 at 0.3 um 242, all fixed (inverted_elements.py: 0), 440 s, 2 GB
#   0 = none
# The .geo refined next to the pores is the one with a "Background Field" (refine() of merope_pore_pressure.py).
HIGH_ORDER_OPTIMIZE_UNIFORM=2
HIGH_ORDER_OPTIMIZE_REFINED=1

# Extra Gmsh arguments.
GMSH_EXTRA_ARGS=(
    # Optional partitioning:
    # -setnumber Mesh.MetisAlgorithm 2
    # -part 50
    # -part_split
)

# =====ERRORS==================================================================
if [[ ! -x "$GMSH" ]]; then
    echo "ERROR: Gmsh executable not found or not executable: $GMSH" >&2
    exit 1
fi

if [[ ! -d "$FOLDER" ]]; then
    echo "ERROR: Input folder not found: $FOLDER" >&2
    exit 1
fi

# =====MESH GENERATION=========================================================
total=0
success=0
failed=0
to_check=0

while IFS= read -r -d '' geo; do
    total=$((total + 1))

    out="${geo%.geo}"
    msh="${out}.msh"
    log="${out}.log"

    echo "============================================================"
    echo "[$total] Processing: $geo"
    echo "    Format: MSH $MSH_FORMAT"

    {
        echo "============================================================"
        echo "Gmsh mesh generation log"
        echo "============================================================"
        echo "Input geometry : $geo"
        echo "Output mesh    : $msh"
        echo "Requested format: MSH $MSH_FORMAT"
        echo "Gmsh executable: $GMSH"
        echo "Start time     : $(date '+%Y-%m-%d %H:%M:%S')"
        echo
    } > "$log"

    # Detect and override Mesh.MshFileVersion in the .geo file.

    if grep -Eq '^[[:space:]]*Mesh[.]MshFileVersion[[:space:]]*=' "$geo"; then
        old_format=$(
            sed -nE \
                's/^[[:space:]]*Mesh[.]MshFileVersion[[:space:]]*=[[:space:]]*([0-9]+([.][0-9]+)?).*/\1/p' \
                "$geo" | head -n 1
        )

        if [[ -n "$old_format" && "$old_format" != "$MSH_FORMAT" ]]; then
            echo "    Override: $old_format -> $MSH_FORMAT"

            {
                echo "Geometry format override:"
                echo "  Original: $old_format"
                echo "  Requested: $MSH_FORMAT"
                echo
            } >> "$log"

            # Replace the numeric version in the assignment.
            if ! sed -i -E \
                "s/^([[:space:]]*Mesh[.]MshFileVersion[[:space:]]*=[[:space:]]*)[0-9]+([.][0-9]+)?/\1${MSH_FORMAT}/" \
                "$geo"; then
                echo "ERROR: Could not update format in $geo" | tee -a "$log"
                failed=$((failed + 1))
                continue
            fi

        elif [[ -n "$old_format" ]]; then
            echo "    Geometry format already matches: $old_format"
            echo "Geometry format: $old_format (already matches)" >> "$log"
        else
            echo "    WARNING: Format assignment found but not recognized."
            echo "WARNING: Could not parse Mesh.MshFileVersion assignment." >> "$log"
        fi
    else
        echo "    No Mesh.MshFileVersion assignment found in .geo."
        echo "Geometry format: no explicit assignment found" >> "$log"
    fi

    # Mesh refined next to the pores (size fields of refine()) or uniform size of Mérope.

    if grep -q '^Background Field' "$geo"; then
        high_order_optimize=$HIGH_ORDER_OPTIMIZE_REFINED
        mesh_kind="$(grep -m 1 -o 'Refinement next to the pores.*' "$geo" | sed 's/ (refine() of merope_pore_pressure.py)//')"
    else
        high_order_optimize=$HIGH_ORDER_OPTIMIZE_UNIFORM
        mesh_kind="uniform size of Mérope"
    fi
    echo "    Mesh: $mesh_kind | HighOrderOptimize $high_order_optimize"
    {
        echo "Mesh: $mesh_kind"
        echo "HighOrderOptimize: $high_order_optimize"
    } >> "$log"

    # Run Gmsh and capture ALL output.

    start_seconds=$(date +%s)

    echo "    Running Gmsh..."

    {
        echo
        echo "Gmsh output:"
        echo "------------------------------------------------------------"
    } >> "$log"

    set +e
    "$GMSH" "$geo" \
        -3 \
        -format "$GMSH_FORMAT" \
        -setnumber Mesh.HighOrderOptimize "$high_order_optimize" \
        "${GMSH_EXTRA_ARGS[@]}" \
        -o "$msh" \
        >> "$log" 2>&1
    status=$?
    set -e

    end_seconds=$(date +%s)
    elapsed=$((end_seconds - start_seconds))

    ok=0
    if [[ $status -eq 0 && -s "$msh" ]]; then
        ok=1
    fi

    # Show only relevant information in the terminal.
    if [[ $ok -eq 1 ]]; then
        echo "    SUCCESS | $(basename "$msh") | ${elapsed}s"
    else
        echo "    FAILED  | $(basename "$geo") | exit code $status"

        # Show warnings and errors immediately when a mesh fails.
        grep -Ei 'warning|error|fatal' "$log" | tail -n 20 || true
    fi

    {
        echo
        echo "------------------------------------------------------------"
        echo "Exit code    : $status"
        echo "Elapsed time : ${elapsed} seconds"
        echo "End time     : $(date '+%Y-%m-%d %H:%M:%S')"
    } >> "$log"

    # Inverted elements (jac. < 0) as Gmsh counts them: before the high-order optimisation, the result of the
    # optimisation (HighOrderOptimize 1) and the count after it (HighOrderOptimize 2). The count of Gmsh is a
    # conservative bound (an element it lists can be valid): a flagged mesh is checked with inverted_elements.py.

    if [[ $ok -eq 1 ]]; then
        before=$(grep -E ': Volume mesh: worst distortion' "$log" | tail -n 1 | sed -E 's/^[A-Za-z]+ *: //' || true)
        result=$(grep -E ': Optimization (succeeded|partially|failed)' "$log" | tail -n 1 | sed -E 's/^[A-Za-z]+ *: //' || true)
        after=$(grep -E ': Final volume mesh: worst distortion' "$log" | tail -n 1 | sed -E 's/^[A-Za-z]+ *: //' || true)
        if [[ -n "$before" ]]; then echo "    Before optimisation: $before"; fi
        if [[ -n "$result" ]]; then echo "    Optimisation       : $result"; fi
        if [[ -n "$after" ]]; then echo "    After optimisation : $after"; fi

        check=0
        if [[ -n "$after" ]]; then
            if [[ "$after" == *"jac. <"* ]]; then check=1; fi
        elif [[ -n "$result" ]]; then
            if [[ "$result" == *failed* ]]; then check=1; fi
        elif [[ "$before" == *"jac. <"* ]]; then
            check=1
        fi
        if [[ $check -eq 1 ]]; then
            echo "    WARNING | possible inverted elements: python3 inverted_elements.py $(dirname "$msh")"
            echo "Possible inverted elements (Gmsh): check with inverted_elements.py" >> "$log"
            to_check=$((to_check + 1))
        fi
    fi

    if [[ $ok -eq 1 ]]; then
        echo "Status: SUCCESS" >> "$log"
        success=$((success + 1))
    else
        echo "Status: FAILED" >> "$log"
        failed=$((failed + 1))
    fi

done < <(find "$FOLDER" -type f -name "mesh_*.geo" -print0 | sort -z)

# =====SUMMARY=================================================================

echo
echo "============================================================"
echo "MESH GENERATION SUMMARY"
echo "============================================================"
echo "Folder          : $FOLDER"
echo "Requested format: MSH $MSH_FORMAT"
echo "Total           : $total"
echo "To check        : $to_check (possible inverted elements for Gmsh: inverted_elements.py)"
echo "Successful      : $success"
echo "Failed          : $failed"
echo "============================================================"

if [[ $failed -gt 0 ]]; then
    exit 1
fi

exit 0