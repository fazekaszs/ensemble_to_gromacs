import numpy as np
from pathlib import Path
from typing import Tuple, Dict

DATA_FOLDER = Path("/rhome/PROTMOD/gadaneczm/GROMACS_KRas_processing/data/KRas-G12C/KRas-G12C_bmrb_abr/")
INPUT_FOLDER = DATA_FOLDER / "pdbs"
RESI_IDX_SHIFT = + 1
# TOP_STRUCTURES = 9000
SCORE_SCALE = 10.0
PEF_DATA_TYPE = Dict[str, Tuple[np.ndarray, np.ndarray]]
FORCE_SCALE = 10.0
PHI_FORCE_CONSTANT = 1.477 * FORCE_SCALE
PSI_FORCE_CONSTANT = 0.530 * FORCE_SCALE
GRO_FILENAME = "kras_g12c_gdp_mg.solv.ions.gro"
TOP_FILENAME = "kras_g12c_gdp_mg.full.top"
