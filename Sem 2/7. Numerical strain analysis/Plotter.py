import pickle
import numpy as np
import matplotlib.pyplot as plt


EV_TO_HZ = 1 / 4.135667696e-15  # eV → Hz


def load_peak_data(pickle_file):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data


def make_scatter_plot(data, is_annihilation=False):
    omega_arr = []
    h_peak_log10 = []
    fwhm = []
    labels = []

    for level_key, entry in data.items():
        if not entry.get("success", False):
            continue

        params = entry.get("parameters", {})
        omega = params.get("omega")
        if omega is None:
            omega = params.get("omega_tr")
        if omega is None:
            omega = params.get("omega_ann")

        h_log = entry.get("h_peak_log10")
        t_fwhm = entry.get("t_fwhm_years")

        if omega is None or not np.isfinite(h_log) or not np.isfinite(t_fwhm):
            continue

        if omega <= 0 or t_fwhm <= 0:
            continue

        omega_arr.append(omega * EV_TO_HZ)  # convert to Hz
        h_peak_log10.append(h_log)
        fwhm.append(t_fwhm)
        
        if is_annihilation:
            latex_label = rf"${level_key}$"
        else:
            level_e, level_g = level_key.split()
            latex_label = rf"${level_e} \rightarrow {level_g}$"
        labels.append(latex_label)

    omega_arr = np.array(omega_arr)
    h_peak_log10 = np.array(h_peak_log10)
    fwhm = np.array(fwhm)

    if len(omega_arr) == 0:
        raise RuntimeError("No valid data points to plot.")

    plot_type = "Annihilation" if is_annihilation else "Transition"

    plt.figure(figsize=(9, 7))

    sc = plt.scatter(
        omega_arr,
        h_peak_log10,
        c=np.log10(fwhm),
        cmap="viridis",
        s=40,
    )

    plt.xscale("log")
    plt.xlabel(r"Frequency $\omega$ [Hz]")

    plt.ylabel(r"$\log_{10}(h_{\mathrm{peak}})$")

    cbar = plt.colorbar(sc)
    cbar.set_label(r"$\log_{10}(\mathrm{FWHM\ [years]})$")

    # --- Add labels next to points ---
    for x, y, label in zip(omega_arr, h_peak_log10, labels):
        plt.text(
            x,
            y,
            label,
            fontsize=9,
            ha="left",
            va="bottom"
        )

    plt.title(f"Peak strain vs frequency ({plot_type}, colour = FWHM)")
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pickle_file = "ann_peak_data_alpha_over_l_0p15.pkl"  # change as needed

    is_annihilation = pickle_file.startswith("ann_")

    data = load_peak_data(pickle_file)
    make_scatter_plot(data, is_annihilation=is_annihilation)