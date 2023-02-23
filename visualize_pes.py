import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from config import INPUT_FOLDER, TOP_STRUCTURES, PES_DATA_TYPE


def main():

    x_values: np.ndarray
    pes_dpes_data: PES_DATA_TYPE

    with open(INPUT_FOLDER / f"../angles_top{TOP_STRUCTURES}.pickle", "rb") as f:
        angle_data = pickle.load(f)

    with open(INPUT_FOLDER / f"../pes_dpes_data_top{TOP_STRUCTURES}.pickle", "rb") as f:
        x_values, pes_dpes_data = pickle.load(f)

    if not os.path.exists(INPUT_FOLDER / f"../pes_figures_top{TOP_STRUCTURES}"):
        os.mkdir(INPUT_FOLDER / f"../pes_figures_top{TOP_STRUCTURES}")

    fig, ax = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.5)

    hist_width = 2
    hist_x = np.arange(-180, 180 + hist_width, hist_width)
    plot_hist_x = (hist_x[:-1] + hist_x[1:]) / 2

    keys = list({key[:-4] for key in pes_dpes_data.keys()})
    keys.sort(key=lambda x: int(x.split("-")[0]))

    for resi_name in keys:

        phi_key = resi_name + " PHI"
        psi_key = resi_name + " PSI"

        phi_pes, dphi_pes = pes_dpes_data[phi_key]
        psi_pes, dpsi_pes = pes_dpes_data[psi_key]

        # Min-max scaling for visual purposes
        phi_pes = (phi_pes - np.min(phi_pes)) / (np.max(phi_pes) - np.min(phi_pes))
        psi_pes = (psi_pes - np.min(psi_pes)) / (np.max(psi_pes) - np.min(psi_pes))

        [axis.cla() for axis in ax.flatten()]

        ax[0, 0].plot(x_values, phi_pes, c="red")
        ax[0, 1].plot(x_values, psi_pes, c="red")

        ax[0, 0].scatter(angle_data[phi_key], np.ones_like(angle_data[phi_key]), alpha=0.3, c="green", marker="|")
        ax[0, 1].scatter(angle_data[psi_key], np.ones_like(angle_data[psi_key]), alpha=0.3, c="green", marker="|")

        hist_phi_y, _ = np.histogram(angle_data[phi_key], bins=hist_x)
        hist_psi_y, _ = np.histogram(angle_data[psi_key], bins=hist_x)

        ax[0, 0].bar(plot_hist_x, hist_phi_y / np.max(hist_phi_y), width=hist_width, alpha=0.5)
        ax[0, 1].bar(plot_hist_x, hist_psi_y / np.max(hist_psi_y), width=hist_width, alpha=0.5)

        ax[0, 0].set_title(resi_name + " PHI")
        ax[0, 1].set_title(resi_name + " PSI")

        ax[1, 0].plot(x_values, dphi_pes, c="blue")
        ax[1, 1].plot(x_values, dpsi_pes, c="blue")

        fig.savefig(INPUT_FOLDER / f"../pes_figures_top{TOP_STRUCTURES}/{resi_name}.png", dpi=300)

        print(f"{resi_name} plotting done...")


if __name__ == "__main__":
    main()
