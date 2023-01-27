import os
import pickle
import numpy as np

from typing import List
from pathlib import Path
from config import INPUT_FOLDER, PES_DATA_TYPE, DATA_FOLDER


def parse_gro(gro_path: Path, keys: List[str]):

    with open(gro_path, "r") as f:
        gro_data = f.read()

    gro_data = gro_data.split("\n")[2:-2]

    gro_data = {
        (int(line[:5]), line[10:15].strip()): int(line[15:20])
        for line in gro_data
    }

    out = dict()
    for resi_name in keys:

        resi_idx = int(resi_name.split("-")[0])

        phi_ids = (
            gro_data[(resi_idx - 1, "C")],
            gro_data[(resi_idx, "N")],
            gro_data[(resi_idx, "CA")],
            gro_data[(resi_idx, "C")],
        )

        psi_ids = (
            gro_data[(resi_idx, "N")],
            gro_data[(resi_idx, "CA")],
            gro_data[(resi_idx, "C")],
            gro_data[(resi_idx + 1, "N")],
        )

        out[resi_name + " PHI"] = phi_ids
        out[resi_name + " PSI"] = psi_ids

    return out


def get_xvg(ref_points: np.ndarray, pes_data: np.ndarray, dpes_data: np.ndarray) -> str:

    out = ""
    for x, y, dy in zip(ref_points, pes_data, dpes_data, strict=True):
        out += f"{x:.0f}\t{y:.10f}\t{-dy:.10f}\n"

    out += f"180\t{pes_data[0]:.10f}\t{-dpes_data[0]:.10f}"

    return out


def main():

    x_values: np.ndarray
    pes_dpes_data: PES_DATA_TYPE

    with open(INPUT_FOLDER / "../pes_dpes_data.pickle", "rb") as f:
        x_values, pes_dpes_data = pickle.load(f)

    keys = list({key[:-4] for key in pes_dpes_data.keys()})
    keys.sort(key=lambda x: int(x.split("-")[0]))
    parse_gro(DATA_FOLDER / "kras_g12c_gdp_mg.em_pr.gro", keys)


if __name__ == "__main__":
    main()
