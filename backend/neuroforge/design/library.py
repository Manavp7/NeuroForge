"""Seed molecule library and mock protein targets for generative design.

Targets are illustrative stand-ins (each with an *ideal* pharmacophore vector + drug-like
property windows). They are NOT real binding-site models.
"""

from __future__ import annotations

import numpy as np

# A small, valid, drug-like seed library (mostly CNS-relevant small molecules).
SEED_SMILES: list[str] = [
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "C1=CC(=C(C=C1CCN)O)O",  # dopamine
    "C1=CC2=C(C=C1O)C(=CN2)CCN",  # serotonin
    "OC(=O)C1=CC=CC=C1O",  # salicylic acid
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "CC(N)Cc1ccccc1",  # amphetamine
    "Clc1ccccc1C2=NCC(=O)Nc3ccc(Cl)cc23",  # benzodiazepine-like
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",  # naproxen
    "CC(C)NCC(O)c1ccc(O)c(O)c1",  # isoprenaline-like
    "c1ccc(cc1)C(=O)Nc1ccccc1",  # benzanilide
    "CC1=CC(=O)CC(C)(C)C1",  # isophorone
    "O=C(O)Cc1ccccc1",  # phenylacetic acid
    "Nc1ccc(cc1)S(=O)(=O)N",  # sulfanilamide
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",  # procainamide-like
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # theophylline-like
    "OCC(O)CO",  # glycerol
    "c1ccncc1",  # pyridine
]


class Target:
    def __init__(
        self,
        target_id: str,
        name: str,
        ideal_pharmacophore: list[float],
        property_windows: dict[str, tuple[float, float]],
    ):
        self.target_id = target_id
        self.name = name
        self.ideal_pharmacophore = np.array(ideal_pharmacophore)
        self.property_windows = property_windows


# Ideal pharmacophore vectors are in the *normalized* space of chem.pharmacophore_vector:
# [MW/500, logP/5, TPSA/140, HBD/5, HBA/10, aromaticRings/4, fracCSP3, rotBonds/10]
TARGETS: dict[str, Target] = {
    "TNF_alpha": Target(
        "TNF_alpha",
        "TNF-α (anti-inflammatory)",
        [0.7, 0.6, 0.6, 0.2, 0.4, 0.5, 0.4, 0.4],
        {"mol_weight": (250, 480), "logp": (1.0, 4.0), "tpsa": (40, 110), "qed": (0.4, 1.0)},
    ),
    "D2": Target(
        "D2",
        "Dopamine D2 receptor (dopaminergic)",
        [0.6, 0.7, 0.3, 0.1, 0.3, 0.5, 0.5, 0.4],
        {"mol_weight": (200, 450), "logp": (1.5, 4.5), "tpsa": (20, 80), "qed": (0.4, 1.0)},
    ),
    "SERT": Target(
        "SERT",
        "Serotonin transporter (serotonergic)",
        [0.6, 0.65, 0.35, 0.1, 0.3, 0.5, 0.45, 0.4],
        {"mol_weight": (220, 450), "logp": (1.5, 4.5), "tpsa": (20, 80), "qed": (0.4, 1.0)},
    ),
    "Nav1_7": Target(
        "Nav1_7",
        "Nav1.7 sodium channel (analgesic)",
        [0.7, 0.55, 0.5, 0.2, 0.4, 0.5, 0.4, 0.5],
        {"mol_weight": (250, 500), "logp": (1.0, 4.0), "tpsa": (40, 120), "qed": (0.35, 1.0)},
    ),
    "GABA_A": Target(
        "GABA_A",
        "GABA-A receptor (anti-seizure)",
        [0.65, 0.6, 0.4, 0.15, 0.35, 0.5, 0.45, 0.35],
        {"mol_weight": (220, 450), "logp": (1.0, 4.0), "tpsa": (30, 100), "qed": (0.4, 1.0)},
    ),
    "COX2": Target(
        "COX2",
        "COX-2 (anti-inflammatory, alt)",
        [0.72, 0.5, 0.62, 0.25, 0.45, 0.5, 0.38, 0.42],
        {"mol_weight": (250, 480), "logp": (1.0, 4.0), "tpsa": (50, 120), "qed": (0.4, 1.0)},
    ),
    "AChE": Target(
        "AChE",
        "Acetylcholinesterase (pro-cognitive)",
        [0.6, 0.65, 0.35, 0.1, 0.35, 0.55, 0.5, 0.45],
        {"mol_weight": (220, 470), "logp": (1.0, 4.5), "tpsa": (20, 90), "qed": (0.4, 1.0)},
    ),
    "NMDA": Target(
        "NMDA",
        "NMDA receptor (neuroprotective)",
        [0.6, 0.55, 0.45, 0.2, 0.4, 0.45, 0.5, 0.4],
        {"mol_weight": (200, 450), "logp": (0.5, 4.0), "tpsa": (30, 110), "qed": (0.35, 1.0)},
    ),
}

# Alternative targets available per construct (first is the primary used by state_to_target).
CONSTRUCT_TARGETS: dict[str, list[str]] = {
    "neuroinflammation": ["TNF_alpha", "COX2"],
    "dopaminergic_deficit": ["D2", "AChE"],
    "serotonergic_deficit": ["SERT"],
    "pain_index": ["Nav1_7", "COX2"],
    "mood_index": ["SERT"],
    "seizure_risk": ["GABA_A", "NMDA"],
}
