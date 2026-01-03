#!/usr/bin/env python3
"""
Compute ground truth stability labels from 20ns simulations.

Extracts f_contact and rmsd_late from 20ns data to create labels.

Usage:
    python compute_ground_truth_labels.py --base-root /path/to/simulations
"""

import numpy as np
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD
import logging

from pipeline.features.mechanical import measure_drift_from_pdbs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def load_existing_keys(output_path: Path):
    """
    Load existing (protein, pocket_id, ligand) keys from CSV if it exists.
    """
    if not output_path.exists():
        return set()

    try:
        df_existing = pd.read_csv(output_path)
        if {"protein", "pocket_id", "ligand_name"}.issubset(df_existing.columns):
            return set(
                zip(
                    df_existing["protein"],
                    df_existing["pocket_id"],
                    df_existing["ligand_name"]
                )
            )
    except Exception as e:
        log.warning(f"Could not load existing CSV for skipping: {e}")

    return set()


def compute_contact_persistence(
    traj_path: Path,
    top_path: Path,
    ligand_resname: str = "UNK"
) -> float:
    """
    Compute f_contact: fraction of frames with ≥1 ligand-protein contact.
    
    Contact defined as heavy-atom distance < 4.0Å
    """
    try:
        u = mda.Universe(str(top_path), str(traj_path))
        
        # Select ligand and protein
        ligand = u.select_atoms(f"resname {ligand_resname}")
        protein = u.select_atoms("protein and not name H*")
        
        if ligand.n_atoms == 0:
            log.warning(f"No ligand found with resname {ligand_resname}")
            return 0.0
        
        # Count frames with contacts
        frames_with_contact = 0
        total_frames = len(u.trajectory)
        
        for ts in u.trajectory:
            # Compute distance matrix
            from MDAnalysis.analysis import distances
            dists = distances.distance_array(
                ligand.positions,
                protein.positions,
                box=u.dimensions
            )
            
            # Check if any distance < 4.0Å
            if np.any(dists < 4.0):
                frames_with_contact += 1
        
        f_contact = frames_with_contact / total_frames
        return f_contact
        
    except Exception as e:
        log.error(f"Failed to compute contact persistence: {e}")
        return 0.0


def compute_rmsd_late(
        traj_path: Path,
        top_path: Path,
        ligand_resname: str = "UNK",
        late_fraction: float = 0.3
) -> float:
    """
    Compute rmsd_late: median ligand RMSD over last fraction of trajectory.
    Returns np.nan if ligand RMSD cannot be computed.
    """
    try:
        u = mda.Universe(str(top_path), str(traj_path))

        # Select ligand heavy atoms
        ligand_sel = f"resname {ligand_resname} and not name H*"
        ligand = u.select_atoms(ligand_sel)

        if ligand.n_atoms == 0:
            log.warning(
                f"No ligand atoms found for RMSD (resname={ligand_resname}); returning NaN"
            )
            return np.nan

        # RMSD with superposition
        rmsd_calc = RMSD(
            u,
            u,
            select=ligand_sel,
            ref_frame=0
        )
        rmsd_calc.run()

        rmsd_values = rmsd_calc.results.rmsd[:, 2]

        n_frames = len(rmsd_values)
        start_idx = int(n_frames * (1 - late_fraction))

        rmsd_late = np.median(rmsd_values[start_idx:])
        return float(rmsd_late)

    except Exception as e:
        log.error(f"Failed to compute RMSD late from {traj_path}: {e}")
        return np.nan

import pandas as pd

def compute_labels_from_plip(plip_summary_path: Path) -> tuple:
    """
    Compute f_contact and n_contacts_20ns from a PLIP summary CSV.

    Supports TWO formats:
    (1) time-resolved frames: frame_2.0ns, frame_4.0ns, ...
    (2) initial/final only: *_initial_frame, *_final_frame
    """
    try:
        df = pd.read_csv(plip_summary_path)

        if df.empty:
            return 0.0, 0

        # ---------- CASE 1: time-resolved PLIP ----------
        if df["Complex"].str.contains(r"frame_\d+\.?\d*ns").any():

            # Extract numeric time
            df["time_ns"] = (
                df["Complex"]
                .str.extract(r"frame_(\d+\.?\d*)ns")[0]
                .astype(float)
            )

            grouped = df.groupby("time_ns")
            timepoints = sorted(grouped.groups.keys())

            if not timepoints:
                return 0.0, 0

            # Earliest timepoint = initial interaction count
            t0 = timepoints[0]
            n_init = len(grouped.get_group(t0))

            if n_init == 0:
                return 0.0, 0

            # Compute interaction-count retention over time
            retention_vals = []
            for t in timepoints:
                n_t = len(grouped.get_group(t))
                retention_vals.append(n_t / n_init)

            f_contact = min(float(np.mean(retention_vals)), 1.0)

            # Contacts specifically at 20 ns
            if 20.0 in grouped.groups:
                n_contacts = len(grouped.get_group(20.0))
            else:
                n_contacts = len(grouped.get_group(timepoints[-1]))

            return f_contact, n_contacts


    # ---------- CASE 2: initial/final only ----------
        else:
            df_init = df[df["Complex"].str.contains("initial_frame")]
            df_final = df[df["Complex"].str.contains("final_frame")]

            base_residues = set(df_init["Residue"].unique())

            if not base_residues:
                return 0.0, 0

            # f_contact = len(base_residues & final_residues) / len(base_residues)
            # n_contacts = len(final_residues)

            n_init = len(df_init)
            n_final = len(df_final)

            if n_init == 0:
                return 0.0, 0

            f_contact = min(n_final / n_init, 1.0)

            n_contacts = n_final

            return float(f_contact), n_contacts

    except Exception as e:
        log.error(f"PLIP ground truth failed: {e}")
        return 0.0, 0



def compute_ground_truth_labels(
    base_root: Path,
    proteins: list,
    output_path: Path,
    ligand_resname: str = "UNK",
    use_plip: bool = True,
    use_drift: bool = True,
    ligand_name : str = 'Milbemycin'
) -> pd.DataFrame:
    """
    Compute ground truth labels for all pockets from 20ns data.
    
    Output columns:
    - protein, pocket_id, replica
    - f_contact_20ns, rmsd_late_20ns
    - label_unstable (0/1)
    - n_contacts_20ns (if using PLIP)
    """
    log.info("="*70)
    log.info("COMPUTING GROUND TRUTH LABELS FROM 20NS DATA")
    log.info("="*70)
    
    rows = []

    existing_keys = load_existing_keys(output_path)
    log.info(f"Found {len(existing_keys)} existing entries to skip")


    for protein in proteins:
        protein_dir = base_root / protein / "simulation_explicit"
        
        if not protein_dir.exists():
            log.warning(f"Skipping {protein} (not found)")
            continue

        for pocket_dir in sorted(protein_dir.glob("pocket*")):
            pocket_id = pocket_dir.name

            key = (protein, pocket_id, ligand_name)
            if key in existing_keys:
                log.info(f"Skipping existing: {protein} {pocket_id} {ligand_name}")
                continue
            replica_dir = pocket_dir / "replica_1"
            
            if not replica_dir.exists():
                continue

            log.info(f"\nProcessing {protein} {pocket_id}")
            
            # Paths
            timescale_dir = replica_dir / "timescale_frames"
            
            if use_plip:
                # Compute from PLIP results
                plip_summary = timescale_dir / "plip_results" / "all_plip_interactions_summary.csv"
                if not timescale_dir.exists():
                    plip_summary = replica_dir / "plip_results" / "all_plip_interactions_summary.csv"


                if not plip_summary.exists():
                    log.warning(f"    No PLIP results")
                    continue
                else:
                    log.info(f"Using PLIP: {plip_summary.name}")
                
                f_contact, n_contacts = compute_labels_from_plip(plip_summary)
                log.info(f"    f_contact (PLIP): {f_contact:.3f} ({n_contacts} contacts)")

                # For RMSD, need trajectory
                top_path = replica_dir / f"{protein}_prepared_{ligand_name}_{pocket_id}_complex_recombined_complex_explicit_initial_frame.pdb"
                traj_path = replica_dir / f"{protein}_prepared_{ligand_name}_{pocket_id}_complex_recombined_complex_explicit_trajectory.dcd"
                reduced_dir = replica_dir / "reduced"
                xtc_path = reduced_dir / "reduced_trajectory.xtc"
                xtc_top = reduced_dir / "reduced_topology.pdb"

                if top_path.exists() and traj_path.exists():
                    rmsd_late = compute_rmsd_late(traj_path, top_path, ligand_resname)
                    log.info(f"    rmsd_late: {rmsd_late:.3f}Å")

                elif xtc_path.exists() and xtc_top.exists():
                    # Reduced XTC case
                    log.info("    Using reduced XTC for RMSD-late")
                    rmsd_late = compute_rmsd_late(
                        traj_path=xtc_path,
                        top_path=xtc_top,
                        ligand_resname=ligand_resname
                    )
                else:
                    log.warning(f"    No trajectory for RMSD computation")
                    rmsd_late = np.nan
                
            else:
                # Compute from MD trajectory directly
                top_path = replica_dir / f"{protein}_prepared_{ligand_name}_{pocket_id}_complex_recombined_complex_explicit_initial_frame.pdb"
                traj_path = replica_dir / f"{protein}_prepared_{ligand_name}_{pocket_id}_complex_recombined_complex_explicit_trajectory.dcd"
                
                if not (top_path.exists() and traj_path.exists()):
                    log.warning(f"    No trajectory files")
                    continue
                
                f_contact = compute_contact_persistence(traj_path, top_path, ligand_resname)
                if not use_drift:
                    rmsd_late = compute_rmsd_late(traj_path, top_path, ligand_resname)
                else:
                    rmsd_late = np.nan
                n_contacts = -1  # Not computed
                
                log.info(f"    f_contact (MD): {f_contact:.3f}")
                log.info(f"    rmsd_late: {rmsd_late:.3f}Å")

            drift = None
            # Apply mechanistic labeling rules
            pdb_initial = replica_dir / f"{protein}_prepared_{ligand_name}_{pocket_id}_complex_recombined_complex_explicit_stripped_initial_frame.pdb"
            pdb_final = replica_dir / f"{protein}_prepared_{ligand_name}_{pocket_id}_complex_recombined_complex_explicit_stripped_final_frame.pdb"
            if Path(pdb_final).is_file():
                try:
                    drift, _, _ = measure_drift_from_pdbs(pdb_initial, pdb_final)
                    print(f"[info] Ligand Drift Distance  : {drift:.3f} ")
                except Exception as e:
                    log.warning(f"[ligand_drift] Error measuring drift: {e}")

            DRIFT_CUTOFF = 6.0  # Å, high-confidence escape
            if ligand_name == "Beauvericin":
                F_CONTACT_CUTOFF = 0.15
                RMSD_CUTOFF = 1.8
            else:
                F_CONTACT_CUTOFF = 0.35
                RMSD_CUTOFF = 1.0
            label_unstable = int(
                (f_contact <= F_CONTACT_CUTOFF)
                or (not np.isnan(rmsd_late) and rmsd_late >= RMSD_CUTOFF)
                or (np.isnan(rmsd_late) and drift is not None and drift >= DRIFT_CUTOFF)
            )
            log.info(f" {protein} {pocket_id}   Label: {'UNSTABLE' if label_unstable else 'STABLE'}")
            
            row = {
                "protein": protein,
                "ligand_name": ligand_name,
                "pocket_id": pocket_id,
                "replica": "replica_1",
                "f_contact_20ns": f_contact,
                "rmsd_late_20ns": rmsd_late,
                "n_contacts_20ns": n_contacts,
                "drift": drift,
                "label_unstable": label_unstable
            }
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("LABELS COMPUTED")
    log.info(f"{'='*70}")
    log.info(f"Total pockets: {len(df)}")
    log.info(f"Unstable: {df['label_unstable'].sum()} ({100*df['label_unstable'].mean():.1f}%)")
    log.info(f"Stable: {(~df['label_unstable'].astype(bool)).sum()} ({100*(1-df['label_unstable'].mean()):.1f}%)")
    
    log.info(f"\nf_contact distribution:")
    log.info(f"  Mean: {df['f_contact_20ns'].mean():.3f}")
    log.info(f"  Median: {df['f_contact_20ns'].median():.3f}")
    log.info(f"  Range: [{df['f_contact_20ns'].min():.3f}, {df['f_contact_20ns'].max():.3f}]")
    
    log.info(f"\nrmsd_late distribution:")
    log.info(f"  Mean: {df['rmsd_late_20ns'].mean():.3f}Å")
    log.info(f"  Median: {df['rmsd_late_20ns'].median():.3f}Å")
    log.info(f"  Range: [{df['rmsd_late_20ns'].min():.3f}, {df['rmsd_late_20ns'].max():.3f}]Å")
    
    # Save
    new_ones = Path("new_drift_20ns.csv")
    df.to_csv(new_ones, index=False)
    log.info(f"\n✓ Saved to: {new_ones}")
    log.info(f"\n✓ You need to copy the new ones to: {output_path}")
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute ground truth stability labels from 20ns data"
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        required=True,
        help="Base directory containing simulations"
    )
    parser.add_argument(
        "--proteins",
        type=str,
        nargs="+",
        default=["AFR1"],
        help="List of protein IDs to process"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labels_all_20ns_back.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--ligand-resname",
        type=str,
        default="UNK",
        help="Ligand residue name (default: UNK)"
    )
    parser.add_argument(
        "--use-plip",
        action="store_true",
        help="Use PLIP results instead of MD trajectory for f_contact"
    )
    
    args = parser.parse_args()
    
    compute_ground_truth_labels(
        base_root=args.base_root,
        proteins=args.proteins,
        output_path=args.output,
        ligand_resname=args.ligand_resname,
        use_plip=args.use_plip,
    )


if __name__ == "__main__":
    # Example usage
    BASE_ROOT = Path("~/simulation_20ns_md/")

    PROTEINS =  [ "AFR1", "ATRF_ASPFU",
                  "CDR1_CANAR", "CDR2_CANAL", "CDR1_CANAR_auris", "CIMG_00780",
                      "CIMG_00533", "CIMG_00780", "CIMG_06197", "CIMG_01418",  "CIMG_09093",
                      "MDR1_CRYNH",  "MDR1_TRIRC", "MDR2_TRIRC", "PDR5_YEAST",
                      "PDH1_CANGA",  "SNQ2_CANGA"
    ]
    #PROTEINS = ["AFR1"]
    OUTPUT = Path("label_drift_20ns.csv")
    
    compute_ground_truth_labels(
        base_root=BASE_ROOT,
        proteins=PROTEINS,
        output_path=OUTPUT,
        ligand_resname="UNK",
        use_plip=True,  # Use PLIP for f_contact
        ligand_name= "Milbemycin"
    )
