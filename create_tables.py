import os
import shutil
import pickle
import numpy as np

from typing import List, Dict, Tuple, Iterable
from pathlib import Path
from config import DATA_FOLDER, INPUT_FOLDER, FORCE_SCALE, SCORE_SCALE, \
    PEF_DATA_TYPE, PHI_FORCE_CONSTANT, PSI_FORCE_CONSTANT, TOP_FILENAME, GRO_FILENAME


def create_id_key(ids: Iterable[int]):
    return frozenset({frozenset(ids[:3]), frozenset(ids[1:])})


def parse_gro(gro_path: Path, keys: List[str]) -> Dict[str, Tuple[int, int, int, int]]:

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


def parse_top(top_path: Path):

    with open(top_path, "r") as f:
        top_data = f.read()

    top_data = top_data.split("\n")

    # Get the index pairs for the boundaries of the dihedral sections.
    dih_header_idxs = [None, ]
    for idx, line in enumerate(top_data):

        if line.startswith("[") and type(dih_header_idxs[-1]) is int:
            dih_header_idxs[-1] = (dih_header_idxs[-1], idx)

        if line.startswith("[ dihedrals ]"):
            dih_header_idxs.append(idx)

    if type(dih_header_idxs[-1]) is int:
        dih_header_idxs[-1] = (dih_header_idxs[-1], len(top_data))

    dih_header_idxs = dih_header_idxs[1:]  # remove dummy None

    # Create the atom id quartets and assign the top file row indices to them.
    dih_ids_to_row_idxs = dict()
    for section_start, section_end in dih_header_idxs:

        section = top_data[section_start + 1:section_end]

        for idx, line in enumerate(section):

            line = line.split(";")[0]
            line = list(filter(lambda x: len(x) != 0, line.split(" ")))

            if len(line) == 0:
                continue

            key = list(map(int, line[:4]))
            # key = tuple(key) if key[0] < key[-1] else tuple(key[::-1])
            key = create_id_key(key)

            dih_ids_to_row_idxs[key] = section_start + 1 + idx

    return top_data, dih_ids_to_row_idxs


def get_xvg(ref_points: np.ndarray, pes_data: np.ndarray, dpes_data: np.ndarray) -> str:

    out = ""
    for x, y, dy in zip(ref_points, pes_data, dpes_data, strict=True):
        out += f"{x:.0f}\t{y:.10f}\t{-dy:.10f}\n"

    out += f"180\t{pes_data[0]:.10f}\t{-dpes_data[0]:.10f}\n"

    return out


def progress_bar(percentage: float, length: int) -> str:

    n_of_hashtags = int(percentage * length)

    out = "["
    out += n_of_hashtags * "#"
    out += (length - n_of_hashtags) * " "
    out += "]"
    out += f" {percentage:.2%}"
    return out


def main():

    x_values: np.ndarray
    pes_dpes_data: PEF_DATA_TYPE

    with open(INPUT_FOLDER / f"../pes_dpes_data_scoreScale{SCORE_SCALE:.0f}.pickle", "rb") as f:
        x_values, pes_dpes_data = pickle.load(f)

    keys = list({key[:-4] for key in pes_dpes_data.keys()})
    keys.sort(key=lambda x: int(x.split("-")[0]))

    resi_to_ids = parse_gro(DATA_FOLDER / GRO_FILENAME, keys)
    top_data, dih_ids_to_row_idxs = parse_top(DATA_FOLDER / TOP_FILENAME)

    if not os.path.exists(INPUT_FOLDER / "../for_gmx"):
        os.mkdir(INPUT_FOLDER / "../for_gmx")

    if not os.path.exists(INPUT_FOLDER / f"../for_gmx/tables_scoreScale{SCORE_SCALE:.0f}"):
        os.mkdir(INPUT_FOLDER / f"../for_gmx/tables_scoreScale{SCORE_SCALE:.0f}")

    print("Writing tables...")

    for table_idx, angle_name in enumerate(resi_to_ids):

        atom_ids_tuple = resi_to_ids[angle_name]

        atom_ids_frset = create_id_key(atom_ids_tuple)
        top_row_idx = dih_ids_to_row_idxs[atom_ids_frset]

        force_constant = PHI_FORCE_CONSTANT if angle_name.endswith(" PHI") else PSI_FORCE_CONSTANT

        new_row = " ".join(map(lambda x: f"{x:5d}", atom_ids_tuple))  # atom ids
        new_row += "     8"  # tabulated dihedral function type
        new_row += f" {table_idx:5d}"  # table index
        new_row += f" {force_constant}"
        new_row += f" ;{top_data[top_row_idx]}"

        top_data[top_row_idx] = new_row

        pes_dpes_table = get_xvg(x_values, *pes_dpes_data[angle_name])

        with open(INPUT_FOLDER / f"../for_gmx/tables_scoreScale{SCORE_SCALE:.0f}/table_d{table_idx}.xvg", "w") as f:
            f.write(pes_dpes_table)

        print("\r", end="")
        print(progress_bar((table_idx+1) / len(resi_to_ids), 30), end=", ")
        print(f"Filename {angle_name} done...", end="")

    top_data = "\n".join(top_data)

    # new_top_filename = f"{TOP_FILENAME[:-9]}_f{FORCE_SCALE:.0f}_sc{SCORE_SCALE:.0f}.new.top"
    new_top_filename = f"{TOP_FILENAME[:-9]}.new.top"
    with open(INPUT_FOLDER / f"../for_gmx/{new_top_filename}", "w") as f:
        f.write(top_data)

    shutil.copy(DATA_FOLDER / GRO_FILENAME, INPUT_FOLDER / "../for_gmx" / GRO_FILENAME)
    shutil.copy(DATA_FOLDER / "../config.py", INPUT_FOLDER / f"../for_gmx/tables_scoreScale{SCORE_SCALE:.0f}/config.py")


if __name__ == "__main__":
    main()
