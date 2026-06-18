import argparse
import ast
import json
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN

from matminer.featurizers.structure import (
    GlobalSymmetryFeatures,
    StructuralHeterogeneity,
    ChemicalOrdering,
    StructuralComplexity,
    RadialDistributionFunction,
    OrbitalFieldMatrix,
)
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    TMetalFraction,
    ValenceOrbital,
)
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.base import MultipleFeaturizer


warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"


DEFAULT_INPUT_CSV = "data/raw/mp_data.csv"
DEFAULT_OUTPUT_FEATURES_CSV = "data/processed/feats_x.csv"
DEFAULT_OUTPUT_LABELS_CSV = "data/processed/labels_y.csv"


BONDI_VDW_RADII = {
    "H": 1.09,
    "He": 1.40,
    "Li": 1.82,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Ne": 1.54,
    "Na": 2.27,
    "Mg": 1.73,
    "Si": 2.10,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Ar": 1.88,
    "K": 2.75,
    "Ni": 1.63,
    "Cu": 1.40,
    "Zn": 1.39,
    "Ga": 1.87,
    "As": 1.85,
    "Se": 1.90,
    "Br": 1.85,
    "Kr": 2.02,
    "Pd": 1.63,
    "Ag": 1.72,
    "Cd": 1.62,
    "In": 1.93,
    "Sn": 2.17,
    "Te": 2.06,
    "I": 1.98,
    "Xe": 2.16,
    "Pt": 1.76,
    "Au": 1.66,
    "Hg": 1.70,
    "Tl": 1.96,
    "Pb": 2.02,
    "U": 1.86,
}


def parse_structure_column(structure_text):
    if pd.isna(structure_text) or structure_text is None:
        return None

    try:
        if isinstance(structure_text, dict):
            return Structure.from_dict(structure_text)

        structure_dict = None

        try:
            clean_text = (
                str(structure_text)
                .replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
            )
            structure_dict = json.loads(clean_text)
        except Exception:
            pass

        if structure_dict is None:
            structure_dict = ast.literal_eval(structure_text)

        if isinstance(structure_dict, dict):
            return Structure.from_dict(structure_dict)

        return None

    except Exception:
        return None


def get_vdw_radius(symbol):
    radius = BONDI_VDW_RADII.get(symbol)

    if radius is not None:
        return radius

    try:
        radius = Element(symbol).van_der_waals_radius
        if radius is not None:
            return float(radius)
    except Exception:
        pass

    return 1.5


def get_vacuum_agnostic_features(structure):
    keys = [
        "t_eff",
        "V_eff",
        "Effective_Density",
        "Effective_VPA",
        "EPF",
        "EAP",
        "Max_EN_Diff",
        "SDR",
        "IBS",
        "AI",
        "EDV",
        "Avg_Bond",
        "Bond_Std",
        "is_centrosymmetric",
        "JAI_EN",
        "JAI_Radius",
    ]

    result = {key: np.nan for key in keys}

    if structure is None or len(structure) == 0:
        return result

    try:
        z_coords = [site.coords[2] for site in structure]

        top_idx = int(np.argmax(z_coords))
        bottom_idx = int(np.argmin(z_coords))

        top_element = structure[top_idx].specie.symbol
        bottom_element = structure[bottom_idx].specie.symbol

        r_top = get_vdw_radius(top_element)
        r_bottom = get_vdw_radius(bottom_element)

        t_eff = (z_coords[top_idx] - z_coords[bottom_idx]) + r_top + r_bottom

        a, b, _ = structure.lattice.abc
        gamma = np.radians(structure.lattice.gamma)
        area_2d = a * b * np.sin(gamma)
        v_eff = area_2d * t_eff

        result["t_eff"] = t_eff
        result["V_eff"] = v_eff

        mass = structure.composition.weight

        result["Effective_Density"] = (mass / v_eff) * 1.660539
        result["Effective_VPA"] = v_eff / structure.num_sites
        result["SDR"] = t_eff / (a + b)
        result["IBS"] = result["Effective_Density"] / t_eff
        result["AI"] = area_2d / t_eff

        total_alpha = 0.0
        total_sphere_volume = 0.0

        for site in structure:
            symbol = site.specie.symbol
            r_vdw = get_vdw_radius(symbol)
            total_alpha += r_vdw**3
            total_sphere_volume += (4.0 / 3.0) * np.pi * (r_vdw**3)

        result["EAP"] = total_alpha / v_eff
        result["EPF"] = total_sphere_volume / v_eff

        composition = structure.composition
        nonmetal_atoms = sum(
            composition[element]
            for element in composition.elements
            if not element.is_metal
        )
        nonmetal_fraction = nonmetal_atoms / composition.num_atoms
        result["EDV"] = v_eff * nonmetal_fraction

        electronegativities = []

        for site in structure:
            value = Element(site.specie.symbol).X
            if value is not None and value > 0:
                electronegativities.append(value)

        if electronegativities:
            result["Max_EN_Diff"] = max(electronegativities) - min(electronegativities)

        crystal_nn = CrystalNN()
        bond_lengths = []

        for i in range(len(structure)):
            try:
                neighbors = crystal_nn.get_nn_info(structure, i)
                for neighbor in neighbors:
                    bond_lengths.append(structure[i].distance(neighbor["site"]))
            except Exception:
                pass

        if bond_lengths:
            result["Avg_Bond"] = float(np.mean(bond_lengths))
            result["Bond_Std"] = float(np.std(bond_lengths))

        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            is_centrosymmetric = 1.0 if analyzer.is_centrosymmetric else 0.0
        except Exception:
            is_centrosymmetric = 0.0

        result["is_centrosymmetric"] = is_centrosymmetric

        symmetry_screening = 0.0 if is_centrosymmetric == 1.0 else 1.0

        top_en = Element(top_element).X
        bottom_en = Element(bottom_element).X

        if top_en is not None and bottom_en is not None:
            result["JAI_EN"] = abs(top_en - bottom_en) * symmetry_screening

        result["JAI_Radius"] = (
            abs(r_top - r_bottom) / max(r_top, r_bottom)
        ) * symmetry_screening

    except Exception:
        pass

    return result


def get_aggregated_site_features(structure):
    if structure is None or len(structure) == 0:
        return {f"CrystalNNF_{i}": np.nan for i in range(61)}

    try:
        fingerprint = CrystalNNFingerprint.from_preset("ops")
        features = [fingerprint.featurize(structure, i) for i in range(len(structure))]
        mean_features = np.nanmean(np.asarray(features, dtype=float), axis=0)
        return {f"CrystalNNF_{i}": mean_features[i] for i in range(61)}
    except Exception:
        return {f"CrystalNNF_{i}": np.nan for i in range(61)}


def add_structure_features(features_df, structures):
    structure_df = pd.DataFrame({"structure": structures})

    featurizers = [
        GlobalSymmetryFeatures(),
        StructuralHeterogeneity(),
        ChemicalOrdering(),
        StructuralComplexity(),
        RadialDistributionFunction(),
        OrbitalFieldMatrix(),
    ]

    for featurizer in featurizers:
        try:
            result = featurizer.featurize_dataframe(
                structure_df.copy(),
                "structure",
                ignore_errors=True,
            )
            features_df = pd.concat(
                [features_df, result.drop(columns=["structure"])],
                axis=1,
            )
        except Exception:
            pass

    return features_df


def add_composition_features(features_df, structures):
    composition_df = pd.DataFrame(
        {
            "composition": [
                structure.composition if structure is not None else None
                for structure in structures
            ]
        }
    )

    featurizer = MultipleFeaturizer(
        [
            ElementProperty.from_preset("magpie"),
            Stoichiometry(),
            TMetalFraction(),
            ValenceOrbital(),
        ]
    )

    composition_features = featurizer.featurize_dataframe(
        composition_df,
        "composition",
        ignore_errors=True,
    )

    features_df = pd.concat(
        [features_df, composition_features.drop(columns=["composition"])],
        axis=1,
    )

    return features_df


def main(args):
    print("Extracting training-set features.")

    raw_df = pd.read_csv(args.input)

    structures = [
        parse_structure_column(row["structure"])
        for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Parsing structures")
    ]

    features_df = raw_df[["material_id"]].copy()

    custom_features = [
        get_vacuum_agnostic_features(structure)
        for structure in tqdm(structures, desc="Physically reconstructed features")
    ]
    features_df = pd.concat([features_df, pd.DataFrame(custom_features)], axis=1)

    site_features = [
        get_aggregated_site_features(structure)
        for structure in tqdm(structures, desc="Site fingerprints")
    ]
    features_df = pd.concat([features_df, pd.DataFrame(site_features)], axis=1)

    features_df = add_structure_features(features_df, structures)
    features_df = add_composition_features(features_df, structures)

    label_columns = ["e_electronic", "e_ionic", "e_total"]
    available_label_columns = [column for column in label_columns if column in raw_df.columns]

    labels_df = raw_df[["material_id"] + available_label_columns].copy()

    os.makedirs(os.path.dirname(args.features_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_output) or ".", exist_ok=True)
    labels_df.to_csv(args.labels_output, index=False)
    features_df.to_csv(args.features_output, index=False)

    print(f"Saved features to {args.features_output}.")
    print(f"Saved labels to {args.labels_output}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--features-output", default=DEFAULT_OUTPUT_FEATURES_CSV)
    parser.add_argument("--labels-output", default=DEFAULT_OUTPUT_LABELS_CSV)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
