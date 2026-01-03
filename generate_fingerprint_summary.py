import os
import glob
import numpy as np
import pandas as pd

TOP_DIR = "~/simulation_20ns_md/"
ligand_name = 'Milbemycin'


# === FEATURE NAME MAP (Fingerprint index → feature name) ===
FEATURE_NAMES = [
    "slope",
    "rmsd_var",
    "mean_disp",
    "var_disp",
    # "f_simple",
    # "f_hbond",
    # "f_residue"
]

rows = []

def load_fingerprints(fp_dir):
    paths = sorted(glob.glob(os.path.join(fp_dir, "fingerprint_*ns.npy")))
    if not paths:
        return None

    fps = {
        os.path.basename(p).split("_")[-1].replace("ns.npy", ""): np.load(p)
        for p in paths
    }
    return fps

# ---- Load 20-ns labels ----
label_file = "label_drift_20ns.csv"
labels_df = pd.read_csv(label_file)

labels_key = {}
for _, r in labels_df.iterrows():
    key = (r['protein'], r['ligand_name'],r['pocket_id'], r['replica'])
    labels_key[key] = {
        "rmsd_20ns": r['rmsd_late_20ns'],
        "f_contact_20ns": r['f_contact_20ns'],
        "ligand_drift":r['drift'],
        "label_unstable": r['label_unstable'],
    }

# ---- Scan proteins and pockets ----
protein_dirs = sorted(glob.glob(os.path.join(TOP_DIR, "*")))

for protein_path in protein_dirs:
    protein = os.path.basename(protein_path)
    pocket_dirs = sorted(glob.glob(os.path.join(protein_path, "simulation_explicit", "pocket*")))

    for pocket_path in pocket_dirs:
        pocket = os.path.basename(pocket_path)
        replica = "replica_1"
        fp_dir = os.path.join(pocket_path, replica, "fingerprints_ca_4")

        if not os.path.isdir(fp_dir):
            continue

        fps = load_fingerprints(fp_dir)
        if fps is None:
            continue

        label_info = labels_key.get((protein, ligand_name, pocket, replica), {
            "rmsd_20ns": np.nan,
            "ligand_drift": np.nan,
            "label_unstable": np.nan
        })

        for t in sorted(fps.keys(), key=lambda x: float(x)):
            vec = fps[t]

            row = {
                "protein": protein,
                "ligand_name" : ligand_name,
                "pocket": pocket,
                "replica": replica,
                "time_ns": float(t),
                "mean": float(vec.mean()),
                "std": float(vec.std()),
                "n_features": 4,
                "rmsd_20ns": label_info["rmsd_20ns"],
                "ligand_drift": label_info["ligand_drift"],
                "f_contact_20ns" : label_info["f_contact_20ns"],
                "label_unstable": label_info["label_unstable"]
            }

             # Fill missing features safely
            for i, name in enumerate(FEATURE_NAMES):
                if i < len(vec):
                    row[name] = float(vec[i])
                else:
                    row[name] = np.nan

            rows.append(row)


# ---- Save final CSV ----
date_tag = "20251225"
df = pd.DataFrame(rows)
file_name = f"dataset/fingerprint_summary_with_components_even_ac_4d_drift_{date_tag}.csv"
df.to_csv(file_name, index=False)
print(f"Saved {file_name}")
