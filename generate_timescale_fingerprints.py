#!/usr/bin/env python3
"""
CA-ONLY timescale fingerprint generation with configurable windowing modes.

This version generates fingerprints WITHOUT PLIP features:
- Fingerprint = [slope, rmsd_var, mean_disp, var_disp, 0, 0, 0]
- First 4 dimensions: CA-only dynamics
- Same shape for compatibility with existing models

Key optimization: Compute RMSD and CA coordinates ONCE for the full trajectory,
then extract timescale-specific fingerprints by slicing the pre-computed arrays.

WINDOW MODES:
=============

1. ACCUMULATED mode (WINDOW_MODE = "accumulated"):
   - Each fingerprint uses data from [0, T]
   - Example at T=10ns: uses frames from 0-10ns (cumulative)
   - Window size grows with T

2. SLIDING mode (WINDOW_MODE = "sliding"):
   - Each fingerprint uses fixed-size window ending at T
   - Example at T=10ns with 8ns window: uses frames from 2-10ns
   - Window size is constant (except for early times)

Configuration:
  - WINDOW_MODE: "accumulated" or "sliding"
  - WINDOW_SIZE_NS: 4.0, 6.0, 8.0, etc. (only used in sliding mode)
  - TIMESCALES: [2, 4, 6, 8, 10, 12, 14, 16, 18]

Examples:
---------
ACCUMULATED mode (WINDOW_MODE="accumulated"):
  T=2ns  → window [0, 2]ns
  T=4ns  → window [0, 4]ns
  T=10ns → window [0, 10]ns
  T=18ns → window [0, 18]ns

SLIDING mode with 8ns window (WINDOW_MODE="sliding", WINDOW_SIZE_NS=8.0):
  T=2ns  → window [0, 2]ns   (< 8ns, so uses [0, T])
  T=4ns  → window [0, 4]ns   (< 8ns, so uses [0, T])
  T=8ns  → window [0, 8]ns   (= 8ns)
  T=10ns → window [2, 10]ns  (8ns sliding window)
  T=14ns → window [6, 14]ns  (8ns sliding window)
  T=18ns → window [10, 18]ns (8ns sliding window)

SLIDING mode with 4ns window (WINDOW_MODE="sliding", WINDOW_SIZE_NS=4.0):
  T=2ns  → window [0, 2]ns
  T=4ns  → window [0, 4]ns
  T=6ns  → window [2, 6]ns   (4ns sliding window)
  T=10ns → window [6, 10]ns  (4ns sliding window)
  T=18ns → window [14, 18]ns (4ns sliding window)
"""

import logging
from pathlib import Path
import numpy as np
from MDAnalysis.analysis import align

from pipeline.sim.processor import SimulationProcessor
from pipeline.config import PROJECT_DIR
from pipeline.io.prankweb import load_prankweb_pocket_residues
from MDAnalysis.analysis.align import rotation_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Timescale endpoints to generate fingerprints for
TIMESCALES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14]

# Window mode: "accumulated" or "sliding"
WINDOW_MODE = "accumulated"

# Window size for sliding mode (ignored in accumulated mode)
WINDOW_SIZE_NS = 4.0  # Options: 4.0, 6.0, 8.0, etc.

# Total simulation time
TOTAL_TIME_NS = 20.0

def extract_aligned_ca_coords(u, sel_str, start_idx, end_idx):
    """
    Stream aligned CA coords for frames [start_idx, end_idx] (inclusive),
    aligned to frame start_idx as reference. Returns np.array (n_frames, n_atoms, 3).
    """
    ca = u.select_atoms(sel_str)
    coords = []

    # Set reference frame
    u.trajectory[start_idx]
    ref = ca.positions.copy()

    for i in range(start_idx, end_idx + 1):
        u.trajectory[i]
        mob = ca.positions.copy()

        R, _ = rotation_matrix(mob, ref)  # best-fit rotation
        mob_aligned = (mob - mob.mean(axis=0)) @ R + ref.mean(axis=0)

        coords.append(mob_aligned.astype(np.float32))

    return np.stack(coords, axis=0)

def run_timescale_fingerprints_ca_only(
        root="~/simulation_20ns_md/",
        ligand_name='Milbemycin'
):
    """Generate CA-only fingerprints for multiple timescales efficiently.

    Strategy:
    1. Load trajectory ONCE per pocket
    2. Align trajectory to reference frame
    3. Compute full RMSD and CA coords ONCE
    4. Generate fingerprints for all timescales by slicing pre-computed data
    5. Output format: [slope, rmsd_var, mean_disp, var_disp, 0, 0, 0]
       - First 4: CA dynamics
       - Last 3: zeros (PLIP disabled)
    """
    root = Path(root)

    for protein_dir in root.iterdir():
        if not protein_dir.is_dir():
            continue

        protein_id = protein_dir.name
            
        for pocket_dir in protein_dir.glob("simulation_explicit/*"):
            replica_dir = pocket_dir / "replica_1"

            if not replica_dir.exists():
                continue

            pocket_id = pocket_dir.name
            out_fpdir = replica_dir / "fingerprints_ca_4"
            out_fpdir.mkdir(exist_ok=True)

            log.info(f"\n{'='*60}")
            log.info(f"Processing {protein_id} / {pocket_id} --> output {out_fpdir}")
            log.info(f"{'='*60}")

            # Early skip if all fingerprints already exist
            expected = [out_fpdir / f"fingerprint_{T}ns.npy" for T in TIMESCALES]
            if all(p.exists() for p in expected):
                log.info(f"⊘ All fingerprints exist, skipping {protein_id}/{pocket_id}")
                continue

            # Check for cached fp4 files and copy them before loading trajectory
            for T in TIMESCALES:
                cached_fp4 = replica_dir / f"fingerprints_ca/fingerprint_{T}ns.npy"
                fp_path = out_fpdir / f"fingerprint_{T}ns.npy"

                if fp_path.exists():
                    log.info(f"⊘ Skipped existing: {fp_path.name}")
                    continue

                if cached_fp4.exists():
                    fp_ac = np.load(cached_fp4)
                    fp = np.zeros(7, dtype=float)
                    fp[:4] = fp_ac[:4]
                    np.save(fp_path, fp)
                    log.info(f"  ✓ Copied cached fp4 → {fp_path.name}: {fp}")
                    continue

            # Determine which timescales actually need computation
            missing_T = []
            for T in TIMESCALES:
                fp_path = out_fpdir / f"fingerprint_{T}ns.npy"
                if not fp_path.exists():
                    missing_T.append(T)

            if not missing_T:
                log.info(f"⊘ Nothing to compute for {protein_id}/{pocket_id}")
                continue

            # ============================================================
            # Initialize processor with PLIP DISABLED
            # ============================================================
            proc = SimulationProcessor(
                replica_dir=replica_dir,
                protein_id=protein_id,
                pocket_id=pocket_id,
                ligand_name=ligand_name,
                logger=log,
                use_timescale_plip=False  # 🔧 PLIP DISABLED
            )

            try:
                # ============================================================
                # STEP 0: Load pocket residues (required for RMSD calculation)
                # ============================================================
                prank_csv = (
                        PROJECT_DIR
                        / "mutation_pipeline"
                        / "outputs"
                        / "prank"
                        / f"{protein_id}_relaxed.pdb_predictions.csv"
                )

                proc.pocket_residues = load_prankweb_pocket_residues(
                    prank_csv, pocket_name=pocket_id
                )

                if not proc.pocket_residues:
                    log.error(f"No pocket residues found for {protein_id}/{pocket_id}")
                    continue

                log.info(f"Loaded {len(proc.pocket_residues)} pocket residues")

                # ============================================================
                # STEP 1: Compute heavy calculations ONCE for full trajectory
                # ============================================================
                log.info("Computing full trajectory metrics (RMSD, CA coords)...")

                # Compute full RMSD series
                full_rmsd = proc._compute_pocket_rmsd_per_frame()
                if full_rmsd is None:
                    log.error(f"Failed to compute RMSD for {protein_id}/{pocket_id}")
                    continue

                # Define pocket CA selection string
                pocket_sel_str = "protein and name CA and resid " + " ".join(
                    map(str, proc.pocket_residues)
                )

                log.info(f"✓ Computed full trajectory: {len(full_rmsd)} frames")
                log.info(f"  RMSD shape: {full_rmsd.shape}")

                # ============================================================
                # STEP 2: Generate fingerprints for each timescale
                # Supports both ACCUMULATED and SLIDING WINDOW modes
                # NO PLIP - output is 7D with last 3 dimensions = 0
                # CA coords are streamed per window (memory safe)
                # ============================================================

                for T in missing_T:
                    fp_path = out_fpdir / f"fingerprint_{T}ns.npy"

                    try:
                        n_frames = len(full_rmsd)

                        # Calculate window based on mode
                        if WINDOW_MODE == "accumulated":
                            # Accumulated mode: always start from 0
                            start_time_ns = 0.0
                            end_time_ns = T
                            mode_label = "accumulated"

                        elif WINDOW_MODE == "sliding":
                            # Sliding window mode: fixed window size
                            if T <= WINDOW_SIZE_NS:
                                # For early times, use [0, T]
                                start_time_ns = 0.0
                                end_time_ns = T
                            else:
                                # For later times, use sliding window [T-window_size, T]
                                start_time_ns = T - WINDOW_SIZE_NS
                                end_time_ns = T
                            mode_label = f"sliding-{WINDOW_SIZE_NS}ns"

                        else:
                            raise ValueError(f"Unknown WINDOW_MODE: {WINDOW_MODE}")

                        # Convert times to frame indices
                        start_frame_idx = int((start_time_ns / TOTAL_TIME_NS) * (n_frames - 1))
                        end_frame_idx = int((end_time_ns / TOTAL_TIME_NS) * (n_frames - 1))

                        # Ensure valid indices
                        start_frame_idx = max(0, start_frame_idx)
                        end_frame_idx = min(end_frame_idx, n_frames - 1)

                        # Slice pre-computed RMSD array for the window
                        rmsd_slice = full_rmsd[start_frame_idx:end_frame_idx + 1]
                        
                        # Stream aligned CA coords ONLY for this window (prevents OOM)
                        ca_coords_slice = extract_aligned_ca_coords(
                            proc.universe, pocket_sel_str, start_frame_idx, end_frame_idx
                        )

                        # Calculate actual window duration
                        actual_window_ns = end_time_ns - start_time_ns

                        # Sanity check: Log CA displacement statistics for this window
                        disp = np.linalg.norm(
                            ca_coords_slice[1:] - ca_coords_slice[:-1], axis=2
                        )

                        log.info(
                            f"  {T}ns ({mode_label}): window [{start_time_ns:.1f}, {end_time_ns:.1f}]ns "
                            f"({actual_window_ns:.1f}ns duration) "
                            f"→ frames {start_frame_idx}-{end_frame_idx} ({len(rmsd_slice)} frames) | "
                            f"CA disp: mean={disp.mean():.3f} Å, max={disp.max():.3f} Å"
                        )

                        # Compute CA-only fingerprint (returns first 4 dimensions)
                        fp4 = proc.compute_mechanistic_fingerprint(
                            rmsd_series=rmsd_slice,
                            ca_coords=ca_coords_slice,
                            time_ns=T,
                            window_start_ns=start_time_ns,
                            window_end_ns=end_time_ns
                        )

                        # Enforce 7D format: [CA-only (4D), zeros for PLIP (3D)]
                        fp = np.zeros(7, dtype=float)
                        fp[:4] = fp4[:4]  # Copy first 4 dimensions
                        # fp[4:7] remain zero

                        # Save fingerprint
                        np.save(fp_path, fp)
                        log.info(f"  ✓ Saved {fp_path.name}: {fp}")

                    except Exception as e:
                        log.error(f"  ✗ Failed at {T}ns: {e}")
                        import traceback
                        log.error(traceback.format_exc())
                        continue

                log.info(f"✓ Completed {protein_id}/{pocket_id}")

            except Exception as e:
                log.error(f"✗ Failed to process {protein_id}/{pocket_id}: {e}")
                import traceback
                log.error(traceback.format_exc())
                continue


if __name__ == "__main__":
    run_timescale_fingerprints_ca_only(
    )
    # Run with: python -m pipeline.training.generate_timescale_fingerprints_ca_only
