import pickle
from typing import Dict
import numpy as np

from config import INPUT_FOLDER

DELTA_ANGLE = 1.0
KERNEL_WIDTH_SCALE = 1


def kde_kernel(distances: np.ndarray, kernel_width: float):

    alpha = 1 + 1 / np.tan(kernel_width * np.pi / 360) ** 2

    kernel_values = np.cos(distances * np.pi / 360) ** alpha

    return kernel_values


def get_pdf(data: np.ndarray) -> np.ndarray:

    ref_points = np.arange(-180, 180, DELTA_ANGLE)

    dmx = np.abs(ref_points[:, np.newaxis] - data[np.newaxis, :])
    mask = dmx > 180
    dmx[mask] = 360 - dmx[mask]

    kernel_width = KERNEL_WIDTH_SCALE * 360 / len(data) ** (1 / 3)
    # kernel_values = np.exp(- 0.5 * (dmx / kernel_width) ** 2)
    kernel_values = kde_kernel(dmx, kernel_width)
    kernel_values = np.sum(kernel_values, axis=1) / len(data)

    return kernel_values


def get_pes(data: np.ndarray) -> np.ndarray:

    # data = data + 1E-5
    # data /= np.sum(data)

    out = -np.log(data)
    out_min = np.min(out)
    out_max = np.max(out)
    out = (out - out_min) / (out_max - out_min)

    return out


def main():

    with open(INPUT_FOLDER / "../angles.pickle", "rb") as f:
        data: Dict[str, np.ndarray] = pickle.load(f)

    keys = list({key[:-4] for key in data.keys()})
    keys.sort(key=lambda x: int(x.split("-")[0]))

    x_values = np.arange(-180, 180, DELTA_ANGLE)

    fit_data = (x_values, dict())

    resi_name: str
    for resi_name in keys:

        phi_key = resi_name + " PHI"
        psi_key = resi_name + " PSI"

        data_phi = data[phi_key]
        data_psi = data[psi_key]

        pdf_phi = get_pdf(data_phi)
        pdf_psi = get_pdf(data_psi)

        pes_phi = get_pes(pdf_phi)
        pes_psi = get_pes(pdf_psi)

        dpes_phi = pes_phi[1:] - pes_phi[:-1]
        dpes_phi = np.append(dpes_phi, (dpes_phi[0] + dpes_phi[-1]) / 2) / DELTA_ANGLE

        dpes_psi = pes_psi[1:] - pes_psi[:-1]
        dpes_psi = np.append(dpes_psi, (dpes_psi[0] + dpes_psi[-1]) / 2) / DELTA_ANGLE

        fit_data[1][phi_key] = (pes_phi, dpes_phi)
        fit_data[1][psi_key] = (pes_psi, dpes_psi)

    with open(INPUT_FOLDER / "../pes_dpes_data.pickle", "wb") as f:
        pickle.dump(fit_data, f)


if __name__ == "__main__":
    main()
