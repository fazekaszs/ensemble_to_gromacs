import pickle
from typing import Dict, List
import numpy as np

from config import INPUT_FOLDER, SCORE_SCALE

DELTA_ANGLE = 1.0
KERNEL_WIDTH_SCALE = 1


def kde_kernel(distances: np.ndarray, kernel_width: float):

    alpha = 1 + 1 / np.tan(kernel_width * np.pi / 360) ** 2

    kernel_values = np.cos(distances * np.pi / 360) ** alpha

    return kernel_values


def get_pdf(data: np.ndarray, scores: np.ndarray, score_scale: float) -> np.ndarray:

    ref_points = np.arange(-180, 180, DELTA_ANGLE)

    dmx = np.abs(ref_points[:, np.newaxis] - data[np.newaxis, :])
    mask = dmx > 180
    dmx[mask] = 360 - dmx[mask]

    weights = np.exp(-scores / score_scale)
    weights /= np.sum(weights)

    kernel_width = KERNEL_WIDTH_SCALE * 360 / len(data) ** (1 / 3)
    kernel_values = kde_kernel(dmx, kernel_width)
    kernel_values = np.sum(kernel_values * weights, axis=1)

    return kernel_values


def get_pef(data: np.ndarray) -> np.ndarray:

    out = -np.log(data)
    out_mean = np.mean(out)
    out_std = np.std(out)
    out = (out - out_mean) / out_std

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

    with open(INPUT_FOLDER / f"../angles_csr.pickle", "rb") as f:
        data: Dict[str, np.ndarray]
        scores: np.ndarray
        data, scores = pickle.load(f)

    keys = list({key[:-4] for key in data.keys()})
    keys.sort(key=lambda x: int(x.split("-")[0]))

    x_values = np.arange(-180, 180, DELTA_ANGLE)

    fit_data = (x_values, dict())

    print("Fitting PEF...")

    resi_name: str
    for counter, resi_name in enumerate(keys):

        phi_key = resi_name + " PHI"
        psi_key = resi_name + " PSI"

        data_phi = data[phi_key]
        data_psi = data[psi_key]

        pdf_phi = get_pdf(data_phi, scores, SCORE_SCALE)
        pdf_psi = get_pdf(data_psi, scores, SCORE_SCALE)

        pef_phi = get_pef(pdf_phi)
        pef_psi = get_pef(pdf_psi)

        dpef_phi = pef_phi[1:] - pef_phi[:-1]
        dpef_phi = np.append(dpef_phi, (dpef_phi[0] + dpef_phi[-1]) / 2) / DELTA_ANGLE

        dpef_psi = pef_psi[1:] - pef_psi[:-1]
        dpef_psi = np.append(dpef_psi, (dpef_psi[0] + dpef_psi[-1]) / 2) / DELTA_ANGLE

        fit_data[1][phi_key] = (pef_phi, dpef_phi)
        fit_data[1][psi_key] = (pef_psi, dpef_psi)

        print("\r", end="")
        print(progress_bar((counter+1) / len(keys), 30), end=", ")
        print(f"{resi_name} is done...", end="")

    with open(INPUT_FOLDER / f"../pef_dpef_data_scoreScale{SCORE_SCALE:.0f}.pickle", "wb") as f:
        pickle.dump(fit_data, f)


if __name__ == "__main__":
    main()
