"""Run MD trajectory analysis and compute graph-level scalars used by the model.

Outputs align with Table \ref{tab:features} (Graph level):
  • Dynamics & geometry (7):
      - pocket_rmsd_per_frame  : per-frame CA RMSD over pocket residues (used as label proxy)
      - ligand_drift           : ligand COM displacement (Å) from initial→final (stripped) PDB
      - rmsd_drift             : mean absolute deviation of complex RMSD across frames (Å)
      - nbd_distance_median    : median COM distance NBD1↔NBD2 (Å)
      - gate_aperture_median   : median COM distance between gating TM helices (Å)
      - cavity_volume          : pocket volume (Å³) via POVME-like grid around PrankWeb center
      - tmh_overlap            : fraction of pocket residues within DeepTMHMM helices [0–1]
  • Energy fluctuations (1): std. dev. of protein–ligand interaction energies (kJ/mol)
  • Domain flags (4): one-hot domain presence (domain_cfg)

Notes:
  - Units are Å unless noted; energy σ is in kJ/mol (from text log).
  - We keep per-frame CA coordinates to build node tensors later.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
from MDAnalysis.analysis import rms
import numpy as np
import MDAnalysis as mda
import pandas as pd
from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.coordinates.PDB import PDBWriter
import os
from pathlib import Path

from pipeline.config import PROJECT_DIR
from pipeline.features.mechanical import calculate_pocket_volume, measure_drift_from_pdbs, \
    compute_contact_stability_frame_phase1
from pipeline.io.prankweb import (
    load_prankweb_pocket_residues,
    load_pocket_center
)
from pipeline.io.gff import parse_deeptmhmm_gff
from pipeline.sim.domain_cfg import DOMAIN_RANGES
import glob
# Add after existing imports (around line 33)
from pipeline.features.mechanical import (
    calculate_pocket_volume,
    measure_drift_from_pdbs,
    compute_contact_stability_phase1  # ← NEW
)

@dataclass
class SimulationProcessor:
    replica_dir: Path
    protein_id: str
    pocket_id: str
    ligand_name: str = "Milbemycin"
    stability_threshold: float = None  #n
    logger: logging.Logger = None
    pocket_residues: list[int] = field(default_factory=list, init=False)
    universe: mda.Universe = field(default=None, init=False)
    target_n : int = 20
    tm_ranges: list = None
    top_path_override_by_frame0: bool = False
    use_timescale_plip: bool = False
    max_timescale_ns: float | None = None
    fingerprintOnly: bool = True

    def __post_init__(self):
        """Initializes logger and loads the Universe after the dataclass is created."""
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

        try:
            top_path = (
                    self.replica_dir
                    / f"{self.protein_id}_prepared_{self.ligand_name}_{self.pocket_id}_complex_recombined_complex_explicit_initial_frame.pdb"
                )
            traj_path = (
                self.replica_dir
                / f"{self.protein_id}_prepared_{self.ligand_name}_{self.pocket_id}_complex_recombined_complex_explicit_trajectory.dcd"
            )

             # ----------------------------------------------------------------------
            # 2. TRY loading with DCD (if exists)
            # ----------------------------------------------------------------------
            if traj_path.exists():
                try:
                    self.logger.info(f"Loading Universe from topology + DCD: {traj_path}")
                    self.universe = mda.Universe(str(top_path), str(traj_path))
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to load DCD, falling back to frame PDBs: {e}")

            # ============================================================
            # Prefer reduced trajectory format if present
            # ============================================================
            reduced_dir = self.replica_dir / "reduced"
            reduced_top = reduced_dir / "reduced_topology.pdb"
            reduced_xtc = reduced_dir / "reduced_trajectory.xtc"

            if reduced_top.exists() and reduced_xtc.exists():
                self.logger.info("Using reduced XTC trajectory")
                self.universe = mda.Universe(
                    str(reduced_top),
                    str(reduced_xtc)
                )
                self.traj_path = traj_path
                return


    # ----------------------------------------------------------------------
            # FALLBACK: Load Universe from frame_*.pdb files
            # ----------------------------------------------------------------------


            frames_dir = self.replica_dir / "frames"
            pdb_files = sorted(glob.glob(str(frames_dir / "frame_*.pdb")))

            if len(pdb_files) == 0:
                raise FileNotFoundError(
                    f"No DCD file found and no frame_*.pdb files in {frames_dir}"
                )
            try:
                self.logger.info(f"Loading Universe from {len(pdb_files)} PDB frames")
                # MDAnalysis loads: Universe(topology, frame1, frame2, ...)
                if self.top_path_override_by_frame0:
                    top_path = pdb_files[0]
                self.universe = mda.Universe(str(top_path), *pdb_files)
                # Now each frame in universe.trajectory corresponds to each PDB

            except Exception as e:
                raise RuntimeError(f"Failed to load Universe from PDB frames: {e}")

            gff_path = (
                    PROJECT_DIR
                    / "mutation_pipeline"
                    / "dataset"
                    / "proteins"
                    / "gff"
                    / f"{self.protein_id}.gff3"
            )

            if gff_path.exists():
                tm_dict = parse_deeptmhmm_gff(gff_path)
                tm_ranges = tm_dict.get(self.protein_id, [])
                self.tm_ranges = tm_ranges
        except Exception as e:
            self.logger.error(f"Failed to load Universe for {self.replica_dir}: {e}")
            self.universe = None

    def run(self, time_ns:Any = None) -> dict[str, Any] | None:
        """Load MD trajectory, calculate all metrics, and return them in a dictionary."""
        if self.universe is None or len(self.universe.trajectory) < 2:
            self.logger.error(
                f"Universe not loaded or trajectory too short for {self.replica_dir}. Skipping."
            )
            return None

        prank_csv = (
            PROJECT_DIR
            / "mutation_pipeline"
            / "outputs"
            / "prank"
            / f"{self.protein_id}_relaxed.pdb_predictions.csv"
        )
        self.pocket_residues = load_prankweb_pocket_residues(
            prank_csv, pocket_name=self.pocket_id
        )
        if not self.pocket_residues:
            self.logger.error(
                f"Could not load pocket residues for {self.pocket_id}. Skipping."
            )
            return None

        domain_cfg = DOMAIN_RANGES.get(self.protein_id, {})
        pocket_rmsd_per_frame = self._compute_pocket_rmsd_per_frame()
        if pocket_rmsd_per_frame is None:
            return None
        ca_coords = [
            self.universe.select_atoms("protein and name CA").positions.copy()
            for ts in self.universe.trajectory
        ]

        features = {}
        # --- 1. Calculate Per-Frame Metrics (as lists) for timescale_frames only ---
        if not self.fingerprintOnly:
            nbd_dists, gate_apertures = self._compute_dynamic_metrics(domain_cfg)

            # --- 2. Calculate Global Metrics (as single values) ---
            ligand_drift = self._compute_ligand_drift_pdb()

            global_rmsd_drift = self._compute_rmsd_drift()
            global_energy_variance = self._compute_energy_variance()
            cavity_volume_per_frame = []
            pocket_center = load_pocket_center(prank_csv, pocket_name=self.pocket_id)
            frames_dir = self.replica_dir / "frames"
            frame_files = sorted(frames_dir.glob("frame_*.pdb"))

            for frame_pdb in frame_files:
                try:
                    vol = calculate_pocket_volume(str(frame_pdb), pocket_center)
                except Exception:
                    vol = 0.0
                cavity_volume_per_frame.append(vol)

            tmh_overlap = self._compute_tmh_overlap(tm_ranges=self.tm_ranges)  # 0-1

            # --- 3. Assemble Final Features Dictionary ---
            features = {
                # Per-frame data
                "pocket_rmsd_per_frame": pocket_rmsd_per_frame,
                "ca_coords": ca_coords,
                # Identity
                "protein_id": self.protein_id,
                "pocket_id": self.pocket_id,

                # Global data (single values)
                "tmh_overlap":         tmh_overlap,
                "rmsd_drift": global_rmsd_drift,
                "energy_variance": global_energy_variance,
                "cavity_volume_per_frame": cavity_volume_per_frame,
                "gate_aperture_median": np.median(gate_apertures)  if gate_apertures else 0.0,
                "nbd_distance_median": np.median(nbd_dists) if nbd_dists else 0.0,
                "ligand_drift":        ligand_drift,
            }

        # ----------------------------------------
        # Select PLIP source depending on mode
        # ----------------------------------------
        if self.use_timescale_plip:
            plip_timescale_dir = self.replica_dir / "timescale_frames"
            plip_csv = plip_timescale_dir / "plip_results/all_plip_interactions_summary.csv"
        else:
            plip_csv = self.replica_dir / "plip_results/all_plip_interactions_summary.csv"
            plip_timescale_dir = None

        fingerprint = self.compute_mechanistic_fingerprint(
            rmsd_series=pocket_rmsd_per_frame,
            ca_coords=ca_coords,
            plip_csv=plip_csv,
            time_ns = time_ns,
            plip_timescale_dir=plip_timescale_dir
        )
        features["mechanistic_fingerprint"] = fingerprint.tolist()
        if not self.fingerprintOnly:
            self._dump_frames()
        return features

    def _compute_dynamic_metrics(
        self, domain_cfg: dict
    ) -> tuple[list[float], list[float]]:
        nbd_dists, gate_apertures = [], []
        if self.tm_ranges is None:
            gff_path = (
                PROJECT_DIR
                / "mutation_pipeline"
                / "dataset"
                / "proteins"
                / "gff"
                / f"{self.protein_id}.gff3"
            )
            tm_ranges = []
            if gff_path.exists():
                tm_dict = parse_deeptmhmm_gff(gff_path)
                tm_ranges = tm_dict.get(self.protein_id, [])
                self.tm_ranges = tm_ranges
        else:
            tm_ranges = self.tm_ranges
        for ts in self.universe.trajectory:
            try:
                nbd1_sel = self.universe.select_atoms(
                    f"resid {domain_cfg['NBD1'][0]}-{domain_cfg['NBD1'][1]} and name CA"
                )
                nbd2_sel = self.universe.select_atoms(
                    f"resid {domain_cfg['NBD2'][0]}-{domain_cfg['NBD2'][1]} and name CA"
                )
                if nbd1_sel.n_atoms > 0 and nbd2_sel.n_atoms > 0:
                    nbd_dists.append(
                        np.linalg.norm(
                            nbd1_sel.center_of_mass() - nbd2_sel.center_of_mass()
                        )
                    )
                else:
                    nbd_dists.append(0.0)
            except (KeyError, IndexError):
                nbd_dists.append(0.0)
            try:
                tm_idx1, tm_idx2 = domain_cfg.get("gate_tm_pair", (None, None))
                if (
                    tm_idx1 is not None
                    and tm_idx1 < len(tm_ranges)
                    and tm_idx2 < len(tm_ranges)
                ):
                    gate1_sel = self.universe.select_atoms(
                        f"resid {tm_ranges[tm_idx1][0]}-{tm_ranges[tm_idx1][1]} and name CA"
                    )
                    gate2_sel = self.universe.select_atoms(
                        f"resid {tm_ranges[tm_idx2][0]}-{tm_ranges[tm_idx2][1]} and name CA"
                    )
                    if gate1_sel.n_atoms > 0 and gate2_sel.n_atoms > 0:
                        gate_apertures.append(
                            np.linalg.norm(
                                gate1_sel.center_of_mass() - gate2_sel.center_of_mass()
                            )
                        )
                    else:
                        gate_apertures.append(0.0)
                else:
                    gate_apertures.append(0.0)
            except (KeyError, IndexError):
                gate_apertures.append(0.0)
        return nbd_dists, gate_apertures

    def _compute_pocket_rmsd_per_frame(self) -> np.ndarray | None:
        try:
            pocket_sel_str = "protein and name CA and resid " + " ".join(
                map(str, self.pocket_residues)
            )
            rmsd_calc = RMSD(
                self.universe, self.universe, select=pocket_sel_str, ref_frame=0
            )
            rmsd_calc.run()
            return rmsd_calc.results.rmsd[:, 2]
        except Exception as e:
            self.logger.error(f"RMSD calculation failed: {e}")
            return None

    def _compute_rmsd_drift(self) -> float | None:
        try:
            ref = self.universe.select_atoms("protein or resname MIL")
            if ref.n_atoms == 0:
                return 0.0
            rmsd_calc = RMSD(self.universe, ref, select="protein or resname MIL")
            rmsd_calc.run()
            rmsd_values = rmsd_calc.results.rmsd[:, 2]
            drift = np.abs(rmsd_values - rmsd_values[0]).mean()
            return float(drift)
        except Exception:
            return 0.0

    def _compute_energy_variance(self) -> float | None:
        try:
            log_path = (
                self.replica_dir
                / f"{self.protein_id}_prepared_{self.ligand_name}_{self.pocket_id}_complex_recombined_complex_interaction_energies.txt"
            )
            if not log_path.exists():
                return 0.0
            energies = np.loadtxt(log_path, skiprows=1, usecols=1, ndmin=1)
            #Physical meaning. σ answers “how much does the interaction energy fluctuate, on average?”—easy to discuss in a paper.

            return float(np.std(energies, ddof=0)) if energies.size > 0 else 0.0
        except Exception:
            return 0.0


    def _compute_ligand_drift_pdb(self) -> float:
        pdb_initial = os.path.join(self.replica_dir, f"{self.protein_id}_prepared_{self.ligand_name}_{self.pocket_id}_complex_recombined_complex_explicit_stripped_initial_frame.pdb")
        pdb_final = os.path.join(self.replica_dir, f"{self.protein_id}_prepared_{self.ligand_name}_{self.pocket_id}_complex_recombined_complex_explicit_stripped_final_frame.pdb")
        if not Path(pdb_final).is_file():
            self.logger.warning(f"[ligand_drift] Missing final frame: {pdb_final}")
            return None
        try:
            drift, _, _ = measure_drift_from_pdbs(pdb_initial, pdb_final)
            print(f"[info] Ligand Drift Distance  : {drift:.3f} ")
            return drift
        except Exception as e:
            self.logger.warning(f"[ligand_drift] Error measuring drift: {e}")
            return None

    def _compute_tmh_overlap(self, tm_ranges: list[tuple[int, int]]) -> float:
        """
        Fraction of pocket residues that fall inside any DeepTMHMM helix.
        Returns 0.0 when either list is empty.
        """
        if not tm_ranges or not self.pocket_residues:
            return 0.0

        tm_set = {resid for start, end in tm_ranges for resid in range(start, end + 1)}
        pocket_set = set(self.pocket_residues)

        overlap = len(pocket_set & tm_set) / len(pocket_set)
        return float(overlap)

    def compute_multiframe_slope_from_snapshots(
            rmsd_series,
            snapshot_times_ns
    ):
        """
        Compute RMSD slope using snapshot-aligned points.
        """
        if len(snapshot_times_ns) < 3:
            return 0.0

        t = np.array(snapshot_times_ns)
        y = np.array(rmsd_series[:len(t)])

        return np.polyfit(t, y, 1)[0]  # Å / ns

    def _rmsd_slope_up_to_ns(
            self,
            rmsd_series: np.ndarray,
            x_ns: float,
            total_ns: float = 20.0,
    ):
        """
        Compute frame-based RMSD slope using all frames from 0 → x_ns,
        but report it as the slope at x_ns.
            """
        rmsd_series = np.asarray(rmsd_series)
        n_frames = len(rmsd_series)

        if n_frames < 3 or x_ns <= 0:
            return 0.0

        # Map x_ns → frame index
        max_frame = int(round(x_ns / total_ns * (n_frames - 1)))
        max_frame = max(2, min(max_frame, n_frames - 1))

        y = rmsd_series[: max_frame + 1]

        # True time axis for these frames
        x = np.linspace(0.0, x_ns, len(y))

        if len(x) < 3:
            return 0.0

        return float(np.polyfit(x, y, 1)[0])

    def _make_times_ns(self):
        if self.max_timescale_ns is None:
            return [2, 4, 6, 8, 10, 12, 14, 16, 18]

        step = 0.5  # or 1.0
        return [t for t in np.arange(step, self.max_timescale_ns + 1e-6, step)]


    def compute_mechanistic_fingerprint(
            self,
            rmsd_series,
            ca_coords,
            time_ns: Any = None,
            plip_csv: Path = None,
            no_plip : bool = True,
            window_start_ns: float | None = None,   # NEW
            window_end_ns: float | None = None      # NEW
    ):
        """
        Compute 7D mechanistic fingerprint.

        Args:
            window_start_ns: Start of the time window (ns) - for slope calculation
            window_end_ns: End of the time window (ns) - for slope calculation
        """
        rmsd_series = np.array(rmsd_series)

        # Compute slope over the actual window
        if window_start_ns is not None and window_end_ns is not None:
            window_duration = window_end_ns - window_start_ns
        else:
            window_duration = time_ns  # Fallback to old behavior

        if len(rmsd_series) >= 3 and window_duration:
            x = np.linspace(0.0, window_duration, len(rmsd_series))
            slope = float(np.polyfit(x, rmsd_series, 1)[0])
        else:
            slope = 0.0

        # 2 — RMSD variance (flexibility)
        rmsd_var = float(np.var(rmsd_series))

        # 5 — Mean displacement per frame (ca_coords dynamics)
        disp = []
        for i in range(1, len(ca_coords)):
            d = np.linalg.norm(ca_coords[i] - ca_coords[i-1], axis=1).mean()
            disp.append(d)
        mean_disp = float(np.mean(disp)) if disp else 0.0

        # 6 — VAR displacement (total relaxation magnitude)
        if len(disp) > 1:
            disp_array = np.array(disp)
            # Use median absolute deviation (more robust to outliers)
            median_disp = np.median(disp_array)
            mad = np.median(np.abs(disp_array - median_disp))

            # Filter outliers (>3 MAD from median)
            mask = np.abs(disp_array - median_disp) < 3 * mad
            clean_disp = disp_array[mask]

            if len(clean_disp) > 1:
                var_disp = float(np.var(clean_disp, ddof=1))
            else:
                var_disp = 0.0
        else:
             var_disp = 0.0
        #var_disp  = float(np.var(disp, ddof=1)) if len(disp) > 1 else 0.0

        # ========== PLIP FEATURES (3D) -  ==========
        if not no_plip and plip_csv and plip_csv.exists():
            try:
                # Option A: original initial/final PLIP (default)
                if not getattr(self, "use_timescale_plip", False):
                    plip_metrics = compute_contact_stability_phase1(plip_csv)
                else:
                    # Option B: summary file with multiple frame_*ns complexes
                    plip_metrics = self._compute_plip_metrics_from_timescale_summary(plip_csv, max_timescale_ns=time_ns)


                if plip_metrics is not None:
                    f_simple = plip_metrics['f_simple']
                    f_hbond = plip_metrics['f_hbond']
                    f_residue = plip_metrics['f_residue']

                    self.logger.info(
                        f"PLIP Phase 1: f_simple={f_simple:.3f}, "
                        f"f_hbond={f_hbond:.3f}, f_residue={f_residue:.3f}"
                    )
                else:
                    # No PLIP data, use neutral values
                    f_simple = 0.5
                    f_hbond = 0.5
                    f_residue = 0.5
                    self.logger.warning("No PLIP data, using default values (0.5)")

            except Exception as e:
                self.logger.error(f"PLIP computation failed: {e}")
                f_simple = 0.5
                f_hbond = 0.5
                f_residue = 0.5
        else:
            # PLIP CSV not provided or doesn't exist
            f_simple = 0.0
            f_hbond = 0.0
            f_residue = 0.0
            self.logger.warning(f"PLIP CSV not found at {plip_csv}, using defaults")

        # ========== COMBINE INTO 10D FINGERPRINT ==========

        fingerprint = np.array([
            # MD features (7)
            slope,
            rmsd_var,
            mean_disp,
            var_disp,
            # PLIP features (3)
            f_simple,
            f_hbond,
            f_residue
        ], dtype=float)

        return fingerprint

    def _dump_frames(self):
        frames_dir = self.replica_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        with PDBWriter(str(frames_dir / "dummy.pdb"), multiframe=False) as W:
            pass  # create writer once to get topology; we'll reopen below

        stride = max(1, len(self.universe.trajectory) // self.target_n)
        for i, ts in enumerate(self.universe.trajectory[::stride]):
            out = frames_dir / f"frame_{i:04d}.pdb"
            if len(list(frames_dir.glob("frame_*.pdb"))) >= self.target_n:
                self.logger.info(f"[SKIP] Already found {self.target_n} frames in {frames_dir}")
                return
            with PDBWriter(str(out), multiframe=False) as W:
                W.write(self.universe.atoms)
            self.logger.info(f"saving {out.stem}")

    # For testing only
    def force_load(self, topology: str, trajectory: str):
        """Forcefully load a custom MD trajectory and topology for testing."""
        try:
            self.universe = mda.Universe(topology, trajectory)
            self.logger.info(f"[TEST] Successfully force-loaded trajectory from {trajectory}")
        except Exception as e:
            self.logger.error(f"[TEST] Failed to force load test trajectory: {e}")
            self.universe = None

    def _compute_plip_metrics_from_timescale_summary(self, plip_csv: Path, max_timescale_ns: Any = None):
        """Compute PLIP contact-retention metrics from a multi-frame summary CSV.

        Expects rows where `Complex` contains substrings like:
            "...frame_2.0ns", "...frame_5.0ns", "...frame_20.0ns"

        We treat 2.0 ns as the reference ("initial") and 20.0 ns as the
        late snapshot ("final"), and reuse the same contact-retention
        definitions as in compute_contact_stability_phase1.
        """
        try:
            df = pd.read_csv(plip_csv)
            # Extract frame time (ns) once
            # Extract frame time (ns) once
            if "frame_ns" not in df.columns:
                df["frame_ns"] = (
                    df["Complex"]
                    .str.extract(r"_frame_([0-9]+(?:\.[0-9]+)?)ns", expand=False)
                    .astype(float)
                )


        except Exception as e:
            self.logger.error(f"Failed to read PLIP summary {plip_csv}: {e}")
            return None

        if "Complex" not in df.columns or "Residue" not in df.columns or "Type" not in df.columns:
            self.logger.warning(
                f"PLIP summary {plip_csv} missing required columns; falling back to defaults"
            )
            return None

         # Initial reference: 0.0 ns
        df_initial = df[df["Complex"].str.contains("frame_0.0ns", na=False)]

        # Final snapshot: current max_timescale_ns
        if max_timescale_ns is None:
            self.logger.warning(
                "max_timescale_ns not set; cannot compute PLIP metrics"
            )
            return None
            # Select the exact frame corresponding to max_timescale_ns
        target_ns = float(max_timescale_ns)
        df_final = df[df["frame_ns"] == target_ns]


        df_initial = (
            df_initial[["Residue", "Type"]]
            .drop_duplicates()
        )

        df_final = (
            df_final[["Residue", "Type"]]
            .drop_duplicates()
        )


        if len(df_initial) == 0 or len(df_final) == 0:
            self.logger.warning(
                f"PLIP summary {plip_csv} lacks frame_2.0ns or frame_20.0ns; cannot compute retention"
            )
            return None

        init_contacts = set(zip(df_initial["Residue"], df_initial["Type"]))
        final_contacts = set(zip(df_final["Residue"], df_final["Type"]))

        # 1) Simple contact retention - based on counts
        n_init = len(init_contacts)
        n_final = len(final_contacts)
        if n_init > 0:
            f_simple = n_final / n_init  # Ratio of final to initial count
        else:
            f_simple = 0.0

        # 2) H-bond retention - based on counts
        init_hb_count = len([t for r, t in init_contacts if t == "HydrogenBond"])
        final_hb_count = len([t for r, t in final_contacts if t == "HydrogenBond"])
        if init_hb_count > 0:
            f_hbond = final_hb_count / init_hb_count
        else:
            f_hbond = 0.0

        # 3) Residue-level persistence - based on counts
        init_res_count = len({r for r, t in init_contacts})
        final_res_count = len({r for r, t in final_contacts})
        if init_res_count > 0:
            f_residue = final_res_count / init_res_count
        else:
            f_residue = 0.0

        return {
            "f_simple": float(f_simple),
            "f_hbond": float(f_hbond),
            "f_residue": float(f_residue),
        }
