import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import DATA_FOLDER, SCORE_SCALE, PEF_DATA_TYPE

hist_color_csr = "cyan"
TAG = ""
LABEL = False


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
    pef_dpef_data: PEF_DATA_TYPE

    with open(DATA_FOLDER / f"angles_csr.pickle", "rb") as f:
        angle_data, _ = pickle.load(f)

    with open(DATA_FOLDER / f"pef_dpef_data_scoreScale{SCORE_SCALE:.0f}.pickle", "rb") as f:
        x_values, pef_dpef_data = pickle.load(f)

    if not os.path.exists(DATA_FOLDER / f"pef_figures_scoreScale{SCORE_SCALE:.0f}{TAG}"):
        os.mkdir(DATA_FOLDER / f"pef_figures_scoreScale{SCORE_SCALE:.0f}{TAG}")

    fig, ax = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.5)
    # plt.subplots_adjust(bottom=0.3)

    pef_patch = mpatches.Patch(color="red", label="PEF [-]")
    single_patches = mpatches.Patch(color="purple", label="single occurrence")
    dpef_patch = mpatches.Patch(color="orange", label="derivative of PEF [-]")
    hist_patch = mpatches.Patch(color=hist_color_csr, label="CS-Rosetta dihedral angle distribution")

    fig.text(0.5, 0.01, "dihedral angle [degree] ", ha="center", fontsize=14)

    hist_width = 2
    hist_x = np.arange(-180, 180 + hist_width, hist_width)
    plot_hist_x = (hist_x[:-1] + hist_x[1:]) / 2

    keys = list({key[:-4] for key in pef_dpef_data.keys()})
    keys.sort(key=lambda x: int(x.split("-")[0]))

    print("Saving pef figures...")

    for counter, resi_name in enumerate(keys):

        phi_key = resi_name + " PHI"
        psi_key = resi_name + " PSI"

        phi_pef, dphi_pef = pef_dpef_data[phi_key]
        psi_pef, dpsi_pef = pef_dpef_data[psi_key]

        # Min-max scaling for visual purposes
        phi_pef = (phi_pef - np.min(phi_pef)) / (np.max(phi_pef) - np.min(phi_pef))
        psi_pef = (psi_pef - np.min(psi_pef)) / (np.max(psi_pef) - np.min(psi_pef))

        [axis.cla() for axis in ax.flatten()]

        ax[0, 0].plot(x_values, phi_pef, c="red")
        ax[0, 1].plot(x_values, psi_pef, c="red")

        ax[0, 0].scatter(angle_data[phi_key], np.ones_like(angle_data[phi_key]), alpha=0.3, c="purple", marker="|")
        ax[0, 1].scatter(angle_data[psi_key], np.ones_like(angle_data[psi_key]), alpha=0.3, c="purple", marker="|")

        hist_phi_y, _ = np.histogram(angle_data[phi_key], bins=hist_x)
        hist_psi_y, _ = np.histogram(angle_data[psi_key], bins=hist_x)

        ax[0, 0].bar(plot_hist_x, hist_phi_y / np.max(hist_phi_y), width=hist_width, color=hist_color_csr, alpha=1)
        ax[0, 1].bar(plot_hist_x, hist_psi_y / np.max(hist_psi_y), width=hist_width, color=hist_color_csr, alpha=1)

        sep_pos = resi_name.find("-")
        resi_name_back = f"{resi_name[-3:]}-{resi_name[:sep_pos]}"
        ax[0, 0].set_title(resi_name_back + " $\mathrm{{\phi}}$", fontsize=15, y=1.05)
        ax[0, 1].set_title(resi_name_back + " $\mathrm{{\psi}}$", fontsize=15, y=1.05)

        ax[1, 0].plot(x_values, dphi_pef, c="orange")
        ax[1, 1].plot(x_values, dpsi_pef, c="orange")

        ax[0, 0].tick_params(axis="both", labelsize=12)
        ax[0, 1].tick_params(axis="both", labelsize=12)
        ax[1, 0].tick_params(axis="both", labelsize=12)
        ax[1, 1].tick_params(axis="both", labelsize=12)

        if LABEL:
            plt.legend(handles=[single_patches, pef_patch, dpef_patch, hist_patch],
                       loc="lower center",
                       bbox_transform=fig.transFigure,
                       ncol=2,
                       bbox_to_anchor=(0.5, 0.05),
                       fontsize=10)

        #plt.show()
        fig.savefig(DATA_FOLDER / f"pef_figures_scoreScale{SCORE_SCALE:.0f}{TAG}/{resi_name}.png", dpi=300)

        print("\r", end="")
        print(progress_bar((counter + 1) / len(keys), 30), end=", ")
        print(f"Filename {resi_name} done...", end="")


if __name__ == "__main__":
    main()
