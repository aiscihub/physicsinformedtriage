"""Domain residue index ranges for each protein (fill as needed)."""
DOMAIN_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    # same for AFR1, we need to fix the protein name using the AFR1_CRYNH
    "AFR1": {
        "NBD1": (222, 474),
        "NBD2": (918, 1160),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = 845–863, TMhelix[11] = 1517–1537
        "ligand_resname": "MIL"
    },
    "ATRF_ASPFU": {
        "NBD1": (197, 439),
        "NBD2": (892, 1130),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = 804–820, TMhelix[11] = 1492–1510
        "ligand_resname": "MIL"
    },
    "CDR1_CANAR": {
        "NBD1": (165, 415),
        "NBD2": (867, 1110),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = 776–793, TMhelix[11] = 1476–1496
        "ligand_resname": "MIL"
    },
    # C. albicans CDR2 transporter (PDR type, 12 TMHs)
    "CDR2_CANAL": {
        "NBD1": (148, 402),
        "NBD2": (857, 1101),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = 763–780, TMhelix[11] = 1465–1485
        "ligand_resname": "MIL"
    },
    # C. auris CDR1 transporter (PDR type, 12 TMHs)
    "CDR1_CANAR_auris": {
        # ✱ Verify these once you map the conserved NBD motifs
        "NBD1": (165, 415),
        "NBD2": (867, 1110),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = 776–793, TMhelix[11] = 1476–1496
        "ligand_resname": "MIL"
    },
    "MDR1_CRYNH": {
        "NBD1": (499, 744),
        "NBD2": (1162, 1402),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = ______, TMhelix[11] = ______
        "ligand_resname": "MIL"
    },
    "MDR1_TRIRC": {
        "NBD1": (167, 432),
        "NBD2": (882, 1125),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = ______, TMhelix[11] = ______
        "ligand_resname": "MIL"
    },
    "MDR2_TRIRC": {
        "NBD1": (422, 667),
        "NBD2": (1086, 1324),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = ______, TMhelix[11] = ______
        "ligand_resname": "MIL"
    },
    "PDH1_CANGA": {
        "NBD1": (153, 409),
        "NBD2": (885, 1128),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = ______, TMhelix[11] = ______
        "ligand_resname": "MIL"
    },
    "PDR5_YEAST": {
        "NBD1": (161, 410),
        "NBD2": (869, 1112),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = ______, TMhelix[11] = ______
        "ligand_resname": "MIL"
    },
    "SNQ2_CANGA": {
        "NBD1": (157, 412),
        "NBD2": (857, 1099),
        #"gate_tm_pair": (5, 11),  # TMhelix[5] = ______, TMhelix[11] = ______
        "ligand_resname": "MIL"
    },
    "CIMG_00533": {
        "NBD1": (172, 330),
        "NBD2": (875, 1026),
        "ligand_resname": "MIL"
    },
    "CIMG_00780": {
        "NBD1": (422, 580),
        "NBD2": (1064, 1215),
        "ligand_resname": "MIL"
    },
    "CIMG_01418": {
        "NBD1": (192, 351),
        "NBD2": (887, 1038),
        "ligand_resname": "MIL"
    },
    "CIMG_06197": {
        "NBD1": (447, 604),
        "NBD2": (1116, 1267),
        "ligand_resname": "MIL"
    },
    "CIMG_09093": {
        "NBD1": (141, 300),
        "NBD2": (833, 983),
        "ligand_resname": "MIL"
    },
}

