import numpy as np
import os
import pickle

from math import atan2, pi
from pathlib import Path
from typing import List

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain

TOP_STRUCTURES = 9000
INPUT_FOLDER = Path("/rhome/PROTMOD/gadaneczm/GROMACS_KRas_processing/KRas-G12C_bmrb_abr/pdbs")


def progress_bar(percentage: float, length: int) -> str:

    n_of_hashtags = int(percentage * length)

    out = "["
    out += n_of_hashtags * "#"
    out += (length - n_of_hashtags) * " "
    out += "]"
    out += f" {percentage:.2%}"
    return out


def get_dihedral(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, r4: np.ndarray) -> float:

    u1: np.ndarray = r2 - r1
    u2: np.ndarray = r3 - r2
    u3: np.ndarray = r4 - r3

    u12: np.ndarray = np.cross(u1, u2)
    u23: np.ndarray = np.cross(u2, u3)

    atan2_arg1 = np.linalg.norm(u2) * np.dot(u1, u23)
    atan2_arg2 = np.dot(u12, u23)

    return atan2(atan2_arg1, atan2_arg2)


def main():

    file_names = os.listdir(INPUT_FOLDER)
    file_names: List[str] = list(filter(lambda x: x.endswith(".pdb"), file_names))
    file_names.sort(key=lambda x: int(x[:-4].split("_")[-1]))
    file_names = file_names[:TOP_STRUCTURES]

    angles_dict = dict()

    for file_idx, file_name in enumerate(file_names):

        kras: Chain = PDBParser(QUIET=True).get_structure("kras", str(INPUT_FOLDER / file_name))[0]["A"]

        for resi_idx in range(1, len(kras) - 1):

            current_resi = kras.child_list[resi_idx]

            prev_c = kras.child_list[resi_idx - 1]["C"]

            curr_n = current_resi["N"]
            curr_ca = current_resi["CA"]
            curr_c = current_resi["C"]

            next_n = kras.child_list[resi_idx + 1]["N"]

            phi = get_dihedral(prev_c.coord, curr_n.coord, curr_ca.coord, curr_c.coord) * 180 / pi
            psi = get_dihedral(curr_n.coord, curr_ca.coord, curr_c.coord, next_n.coord) * 180 / pi

            phi_key = f"{current_resi.get_segid()}-{current_resi.resname} PHI"
            psi_key = f"{current_resi.get_segid()}-{current_resi.resname} PSI"

            if phi_key in angles_dict:
                angles_dict[phi_key].append(phi)
            else:
                angles_dict[phi_key] = [phi, ]

            if psi_key in angles_dict:
                angles_dict[psi_key].append(psi)
            else:
                angles_dict[psi_key] = [psi, ]

        print("\r", end="")
        print(progress_bar(file_idx / len(file_names), 30), end=", ")
        print(f"Filename {file_name} done...", end="")

    angles_dict = {key: np.array(value) for key, value in angles_dict.items()}

    with open(INPUT_FOLDER / "../angles.pickle", "wb") as f:
        pickle.dump(angles_dict, f)


if __name__ == '__main__':
    main()
