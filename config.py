import numpy as np
from pathlib import Path
from typing import Tuple, Dict

DATA_FOLDER = Path("./data")
INPUT_FOLDER = DATA_FOLDER / "KRas-G12C_bmrb_abr/pdbs"
RESI_IDX_SHIFT = + 1
PES_DATA_TYPE = Dict[str, Tuple[np.ndarray, np.ndarray]]
FORCE_CONSTANT = 1.0
GRO_FILENAME = "kras_g12c_gdp_mg.em_pr.gro"
TOP_FILENAME = "kras_g12c_gdp_mg.full.top"
