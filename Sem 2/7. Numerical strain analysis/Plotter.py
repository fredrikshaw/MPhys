import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


EV_TO_HZ = 1 / 4.135667696e-15  # eV → Hz


# -------------------------------------------------------------------
# USER CONTROLS
# -------------------------------------------------------------------

# Points to completely remove from the plot.
# Use the raw pickle keys, e.g. "7g 5g" for transitions or "5g" for annihilations.
EXCLUDE_POINTS = {
    "13k 8k",
    "7p 6p",
     "8g 7g",
}

# Manual label offsets in points.
# Positive dx = right, negative dx = left.
# Positive dy = up, negative dy = down.
LABEL_OFFSETS = {
    "8g 6g": (0, -10),
    "9k 8k": (0, 10),
    # "7g 6g": (5, 0),
    "7h 6h": (-2.5, 2.5),
    "7d 5d": (0, -10),
    "8p 4p": (-5,0),
    # "6f 4f": (-2.5,2),
    # "7g 5g": (10, 12),
    "7p 2p": (0, -10),
    "5d 4d": (0, -10),
    "7f 5f": (0, -10),
    "6f 4f": (-10, 3),
    "8f": (10, -12.5),
    "7f": (10, -10),
    "6f": (10, -7.5),
    "5f": (10, -5),
    "8g": (10, -14.5),
    "7g": (10, -11),
    "5g": (10, -5),
    "6g": (10, 2.5),
    "6h": (10, -10),
    "8h": (10, -7.5),
}

# Points to keep in the scatter plot but hide their labels.
# Use the raw pickle keys, e.g. "7g 5g" for transitions or "5g" for annihilations.
HIDE_LABELS = {
    #"7h 6h",
    #"7p 3p",
    #"6d 5d",
    #"9k 8k",
    # "5g",
}

DEFAULT_LABEL_OFFSET = (10, 3)

# Colour points by l quantum number instead of FWHM.
# Set to True to use discrete l-based colours with no colourbar or legend.
COLOR_BY_L = True

# Save options
SAVE_PLOT = True
PLOTS_SUBFOLDER = "Plots"


def load_peak_data(pickle_file):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data


def make_label(level_key, is_annihilation=False):
    if is_annihilation:
        return rf"${level_key}$"

    level_e, level_g = level_key.split()
    return rf"${level_e} \rightarrow {level_g}$"


def make_output_filename(pickle_file, is_annihilation=False):
    stem = Path(pickle_file).stem
    prefix = "annihilation" if is_annihilation else "transition"
    return f"{prefix}_{stem}.pdf"


import matplotlib.ticker as ticker


def improve_log_x_ticks(ax):
    xmin, xmax = ax.get_xlim()

    major_locator = ticker.LogLocator(base=10.0, numticks=20)
    minor_locator = ticker.LogLocator(
        base=10.0,
        subs=np.arange(2, 10),
        numticks=100,
    )

    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)

    major_ticks = major_locator.tick_values(xmin, xmax)
    major_ticks = major_ticks[(major_ticks >= xmin) & (major_ticks <= xmax)]

    if len(major_ticks) == 0:
        exponent = int(np.floor(np.log10((xmin + xmax) / 2)))
        major_ticks = np.array([10.0 ** exponent])
        return

    if len(major_ticks) <= 1:
        exponent = int(np.floor(np.log10(major_ticks[0])))

        # Major tick becomes "1"
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, pos: rf"${x / 10**exponent:g}$" if x > 0 else ""
            )
        )

        def mantissa_formatter(x, pos):
            if x <= 0:
                return ""

            val = x / 10**exponent

            if 0.1 <= val < 10:
                return rf"${val:g}$"

            return ""

        ax.xaxis.set_minor_formatter(
            ticker.FuncFormatter(mantissa_formatter)
        )

        ax.text(
            1.01,
            -0.06,
            rf"$\times 10^{{{exponent}}}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
        )

    else:
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.tick_params(axis="x", which="minor", labelsize=11)


def make_scatter_plot(
    data,
    is_annihilation=False,
    exclude_points=None,
    hide_labels=None,
    label_offsets=None,
    default_label_offset=DEFAULT_LABEL_OFFSET,
    save_plot=False,
    output_filename=None,
    plots_subfolder=PLOTS_SUBFOLDER,
    color_by_l=False,
):
    if exclude_points is None:
        exclude_points = set()

    if label_offsets is None:
        label_offsets = {}

    if hide_labels is None:
        hide_labels = set()

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18,
    })

    omega_arr = []
    h_peak_log10 = []
    fwhm = []
    l_quantum = []
    labels = []
    keys = []

    for level_key, entry in data.items():
        if level_key in exclude_points:
            continue

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

        keys.append(level_key)
        omega_arr.append(omega * EV_TO_HZ)
        h_peak_log10.append(h_log)
        fwhm.append(t_fwhm)
        l_quantum.append(int(params.get('l_g', params.get('l_e', 0))))
        labels.append(make_label(level_key, is_annihilation=is_annihilation))

    omega_arr = np.array(omega_arr)
    h_peak_log10 = np.array(h_peak_log10)
    fwhm = np.array(fwhm)
    l_quantum = np.array(l_quantum)

    if len(omega_arr) == 0:
        raise RuntimeError("No valid data points to plot.")

    plot_type = "Annihilation" if is_annihilation else "Transition"

    fig, ax = plt.subplots(figsize=(5, 4))

    if color_by_l:
        unique_ls = sorted(set(l_quantum))
        l_to_letter = {
            0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h',
            6: 'i', 7: 'k', 8: 'l', 9: 'm', 10: 'n', 11: 'o', 12: 'q'
        }
        cmap_discrete = plt.cm.get_cmap('tab10', len(unique_ls))
        l_to_idx = {l: i for i, l in enumerate(unique_ls)}
        point_colors = [cmap_discrete(l_to_idx[l]) for l in l_quantum]
        sc = ax.scatter(
            omega_arr,
            h_peak_log10,
            c=point_colors,
            s=40,
        )
    else:
        sc = ax.scatter(
            omega_arr,
            h_peak_log10,
            c=np.log10(fwhm),
            cmap="viridis",
            s=40,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$\log_{10}(\mathrm{FWHM\ [years]})$")

    ax.set_xscale("log")
    ax.set_xlabel(r"Frequency $\omega$ [Hz]")
    ax.set_ylabel(r"$\log_{10}(h_{\mathrm{peak}})$")

    # Add some breathing room so labels do not leave the axes.
    ax.margins(x=0.15, y=0.15)
    improve_log_x_ticks(ax)

    # --- Add labels with manual offsets ---
    for x, y, label, key in zip(omega_arr, h_peak_log10, labels, keys):
        if key in hide_labels:
            continue

        dx, dy = label_offsets.get(key, default_label_offset)

        ha = "center"
        va = "bottom"

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            ha=ha,
            va=va,
            # bbox=dict(
            #     facecolor="white",
            #     edgecolor="none",
            #     alpha=0.7,
            #     pad=0.2,
            # ),
        )

    # ax.set_title(f"Peak strain vs frequency ({plot_type}, colour = FWHM)")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()

    if save_plot:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / plots_subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_filename is None:
            output_filename = f"{plot_type.lower()}_peak_strain.pdf"

        output_path = output_dir / output_filename
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    plt.show()


if __name__ == "__main__":
    pickle_file = "transition_peak_data.pkl"

    is_annihilation = pickle_file.startswith("ann_")

    data = load_peak_data(pickle_file)

    make_scatter_plot(
        data,
        is_annihilation=is_annihilation,
        exclude_points=EXCLUDE_POINTS,
        hide_labels=HIDE_LABELS,
        label_offsets=LABEL_OFFSETS,
        save_plot=SAVE_PLOT,
        output_filename=make_output_filename(pickle_file, is_annihilation=is_annihilation),
        color_by_l=COLOR_BY_L,
    )