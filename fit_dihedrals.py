import pickle
import os
from typing import Dict
from scipy.fft import rfft
import numpy as np
import matplotlib.pyplot as plt

from save_dihedrals import INPUT_FOLDER

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


def get_fft(data: np.ndarray, n_of_waves):

    # https://docs.scipy.org/doc/scipy/tutorial/fft.html

    x_values = np.arange(0, 360, DELTA_ANGLE)

    data_rfft = rfft(data)

    cos_coeffs = 2 * np.real(data_rfft) / len(data)
    sin_coeffs = 2 * np.imag(data_rfft) / len(data)

    cos_mask = np.argsort(np.abs(cos_coeffs))[-1:-n_of_waves-1:-1]
    sin_mask = np.argsort(np.abs(sin_coeffs))[-1:-n_of_waves-1:-1]

    cos_coeffs = cos_coeffs[cos_mask]
    sin_coeffs = sin_coeffs[sin_mask]

    cos_wave = - cos_mask[:, np.newaxis] * x_values[np.newaxis, :] * (np.pi / 180)
    cos_wave = cos_coeffs[:, np.newaxis] * np.cos(cos_wave)

    sin_wave = - sin_mask[:, np.newaxis] * x_values[np.newaxis, :] * (np.pi / 180)
    sin_wave = sin_coeffs[:, np.newaxis] * np.sin(sin_wave)

    wave = np.sum(cos_wave + sin_wave, axis=0)

    return wave


def main():

    fig, ax = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.5)

    hist_width = 2
    hist_x = np.arange(-180, 180 + hist_width, hist_width)
    plot_hist_x = (hist_x[:-1] + hist_x[1:]) / 2

    with open(INPUT_FOLDER / "../angles.pickle", "rb") as f:
        data: Dict[str, np.ndarray] = pickle.load(f)

    keys = list({key[:-4] for key in data.keys()})
    keys.sort(key=lambda x: int(x.split("-")[0]))

    x_values = np.arange(-180, 180, DELTA_ANGLE)

    if not os.path.exists(INPUT_FOLDER / "../pes_figures"):
        os.mkdir(INPUT_FOLDER / "../pes_figures")

    resi_name: str
    for resi_name in keys:

        data_phi = data[resi_name + " PHI"]
        data_psi = data[resi_name + " PSI"]

        pdf_phi = get_pdf(data_phi)
        pdf_psi = get_pdf(data_psi)

        pes_phi = get_pes(pdf_phi)
        pes_psi = get_pes(pdf_psi)

        dpes_phi = pes_phi[1:] - pes_phi[:-1]
        dpes_phi = np.append(dpes_phi, (dpes_phi[0] + dpes_phi[-1]) / 2) / DELTA_ANGLE

        dpes_psi = pes_psi[1:] - pes_psi[:-1]
        dpes_psi = np.append(dpes_psi, (dpes_psi[0] + dpes_psi[-1]) / 2) / DELTA_ANGLE

        # pes_phi_fft = get_fft(pes_phi, n_of_waves=10)
        # pes_psi_fft = get_fft(pes_psi, n_of_waves=10)

        # pes_phi_fft -= np.min(pes_phi_fft) - np.min(pdf_phi)
        # pes_psi_fft -= np.min(pes_psi_fft) - np.min(pdf_psi)

        # Plotting

        [axis.cla() for axis in ax.flatten()]

        ax[0, 0].plot(x_values, pes_phi, c="red")
        ax[0, 1].plot(x_values, pes_psi, c="red")

        # ax[0].plot(x_values, pes_phi_fft, c="grey", alpha=0.5)
        # ax[1].plot(x_values, pes_psi_fft, c="grey", alpha=0.5)

        ax[0, 0].scatter(data_phi, np.ones_like(data_phi), alpha=0.3, c="green", marker="|")
        ax[0, 1].scatter(data_psi, np.ones_like(data_psi), alpha=0.3, c="green", marker="|")

        hist_phi_y, _ = np.histogram(data_phi, bins=hist_x)
        hist_psi_y, _ = np.histogram(data_psi, bins=hist_x)
        ax[0, 0].bar(plot_hist_x, hist_phi_y / np.max(hist_phi_y), width=hist_width, alpha=0.5)
        ax[0, 1].bar(plot_hist_x, hist_psi_y / np.max(hist_psi_y), width=hist_width, alpha=0.5)

        ax[0, 0].set_title(resi_name + " PHI")
        ax[0, 1].set_title(resi_name + " PSI")

        ax[1, 0].plot(x_values, dpes_phi, c="blue")
        ax[1, 1].plot(x_values, dpes_psi, c="blue")

        fig.savefig(INPUT_FOLDER / f"../pes_figures/{resi_name}.png", dpi=300)

        print(f"{resi_name} done...")


if __name__ == "__main__":
    main()
