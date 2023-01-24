import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from save_dihedrals import INPUT_FOLDER


def tick_shifter(tick):

    return f"{(180 / np.pi) * (tick if tick < np.pi else tick - 2 * np.pi):.0f}"


def main():

    with open(INPUT_FOLDER / "../angles.pickle", "rb") as f:
        data: Dict[str, np.ndarray] = pickle.load(f)

    hist_width = 2
    hist_x = np.arange(-180, 180 + hist_width, hist_width)
    plot_hist_x = (np.pi / 180) * (hist_x[:-1] + hist_x[1:]) / 2

    fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection="polar"))
    plt.subplots_adjust(wspace=0.5)

    keys = list({key[:-4] for key in data.keys()})
    keys.sort(key=lambda x: int(x.split("-")[0]))

    if not os.path.exists(INPUT_FOLDER / "../angle_figures"):
        os.mkdir(INPUT_FOLDER / "../angle_figures")

    for resi_name in keys:

        phi_values = data[resi_name + " PHI"]
        psi_values = data[resi_name + " PSI"]

        hist_phi, _ = np.histogram(phi_values, bins=hist_x)
        hist_psi, _ = np.histogram(psi_values, bins=hist_x)

        [axis.cla() for axis in ax]

        x0_ticks = ax[0].get_xticks()
        ax[0].set_xticks(x0_ticks, labels=list(map(tick_shifter, x0_ticks)))

        x1_ticks = ax[1].get_xticks()
        ax[1].set_xticks(x1_ticks, labels=list(map(tick_shifter, x1_ticks)))

        ax[0].set_title(resi_name + " PHI")
        ax[0].bar(plot_hist_x, hist_phi / np.max(hist_phi),
                  width=hist_width * np.pi / 180,
                  bottom=1)

        ax[1].set_title(resi_name + " PSI")
        ax[1].bar(plot_hist_x, hist_psi / np.max(hist_psi),
                  width=hist_width * np.pi / 180,
                  bottom=1)

        ax[0].set_yticks(ax[0].get_yticks(), labels=["" for _ in ax[0].get_yticks()])
        ax[1].set_yticks(ax[1].get_yticks(), labels=["" for _ in ax[1].get_yticks()])

        fig.savefig(INPUT_FOLDER / f"../angle_figures/{resi_name}.png", dpi=300)

        print(f"{resi_name} done...")


if __name__ == "__main__":
    main()
