import numpy as np
from pathlib import Path
from typing import Tuple, Dict

DATA_FOLDER = Path("./data")
INPUT_FOLDER = DATA_FOLDER / "KRas-G12C-MF_abr/pdbs"
RESI_IDX_SHIFT = + 1
TOP_STRUCTURES = 9000
PES_DATA_TYPE = Dict[str, Tuple[np.ndarray, np.ndarray]]
PHI_FORCE_CONSTANT = 1.477 * 10
PSI_FORCE_CONSTANT = 0.530 * 10
GRO_FILENAME = "kras_g12c_empty.em_pr.gro"
TOP_FILENAME = "kras_g12c_empty.full.top"
