import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import constants

SOLAR_MASS = 1.988e30  # [kg]


def calc_clmn(l: int, m, n, a, r_g, mu_a):
    num_1 = (2 ** (4 * l + 4)) * math.factorial(2 * l + n + 1)
    denom_1 = (l + n + 1) ** (2 * l + 4) * math.factorial(n)
    first = num_1 / denom_1

    num_2 = math.factorial(l)
    denom_2 = math.factorial(2 * l) * math.factorial(2 * l + 1)
    second = (num_2 / denom_2) ** 2

    prod = 1.0
    r_plus = calc_r_plus(r_g, a)
    w_plus = calc_w_plus(r_g, a)

    for j in range(1, l + 1):
        term = (j ** 2) * (1 - a ** 2 / r_g ** 2) + 4 * r_plus ** 2 * (m * w_plus - mu_a) ** 2
        prod *= term

    return first * second * prod


def calc_r_plus(r_g, a):
    return r_g + np.sqrt(r_g ** 2 - a ** 2)


def calc_w_plus(r_g, a):
    first = 1 / (2 * r_g)
    num = a / r_g
    denom = 1 + np.sqrt(1 - (a / r_g) ** 2)
    second = num / denom
    return first * second


def calc_alpha(mu_a, r_g):
    return mu_a * r_g

def calc_gamma(l, m, n, a, r_g, mu_a):
    alpha = calc_alpha(mu_a, r_g)
    r_plus = calc_r_plus(r_g, a)
    w_plus = calc_w_plus(r_g, a)
    C_lmn = calc_clmn(l, m, n, a, r_g, mu_a)
    return 2 * mu_a * alpha ** (4 * l + 4) * r_plus * (m * w_plus - mu_a) * C_lmn


def compute_superradiance_data(blackholemass: float):
    """Compute arrays for plotting superradiance growth rates."""
    l_values = [1, 2, 3, 4, 5]
    spins = [0.90, 0.99, 0.999]
    alpha_vals = np.logspace(-2, 1, 500)

    # --- Convert BH mass ---
    m_bh_J = blackholemass * SOLAR_MASS * constants.c ** 2       # [J]
    m_bh_ev = m_bh_J / constants.e                               # [eV]
    G_N = 6.708e-57                                              # [eV^-2]
    r_g = G_N * m_bh_ev                                          # [eV^-1]
    r_g_SI = constants.G * blackholemass * SOLAR_MASS / (constants.c)**2     # [m]

    # --- Print physical parameters in table format ---
    print("\n" + "="*65)
    print(f"{'BLACK HOLE PARAMETERS':^65}")
    print("="*65)
    print(f"{'Quantity':<30}{'Symbol':<10}{'Value':>25}")
    print("-"*65)
    print(f"{'Black hole mass':<30}{'M_BH':<10}{blackholemass:>12.3e}  [M☉]")
    print(f"{'':<30}{'':<10}{m_bh_ev:>12.3e}  [eV]")
    print("-"*65)
    print(f"{'Gravitational radius':<30}{'r_g':<10}{r_g:>12.3e}  [eV⁻¹]")
    print(f"{'':<30}{'':<10}{r_g_SI:>12.3e}  [m]")
    print("="*65 + "\n")

    data = []
    for a_star in spins:
        a = a_star * r_g
        omega_plus = calc_w_plus(r_g, a)

        for l in l_values:
            m = l
            n = l + 1

            mu_vals, gamma_si, gamma_rg = [], [], []

            for alpha in alpha_vals:
                mu_a = alpha / r_g
                if m * omega_plus > mu_a:
                    try:
                        # Gamma is in units of [eV] (rate in natural units)
                        gamma = calc_gamma(l, m, n, a, r_g, mu_a) # [eV]
                        if gamma > 0 and np.isfinite(gamma):
                            gamma_si.append((1 / gamma) * 2.086e-23 * 3.154e7)  # [seconds]
                            gamma_rg.append(gamma * r_g) # [dimensionless]
                            mu_vals.append(mu_a) # [eV]
                    except (OverflowError, ValueError, ZeroDivisionError):
                        pass

            data.append({
                "l": l,
                "a_star": a_star,
                "mu_vals": np.array(mu_vals),
                "gamma_si": np.array(gamma_si),
                "gamma_rg": np.array(gamma_rg),
                "r_g": r_g,
                "bh_mass": blackholemass,
            })
 
    return data



def plot_superradiance_data(data):
    """Plot the superradiance rates with dual y-axes and BH mass annotation."""
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Extract BH mass (solar masses) and gravitational radius
    bh_mass = data[0]["bh_mass"]
    r_g = data[0]["r_g"]

    # Plotting style
    colors = ['blue', 'orange', 'red', 'green', 'purple']
    linestyles = ["--", "-.", "-"]

    l_values = sorted(list({d["l"] for d in data}))
    spins = sorted(list({d["a_star"] for d in data}))

    # --- Plot data ---
    for d in data:
        l_idx = l_values.index(d["l"])
        spin_idx = spins.index(d["a_star"])

        # Plot Gamma^{-1} [years] on the left axis
        ax1.semilogy(d["mu_vals"], d["gamma_si"],
                     color=colors[l_idx % len(colors)],
                     linestyle=linestyles[spin_idx % len(linestyles)],
                     linewidth=2)

    # --- Axis configuration ---
    ax1.set_xlabel(r'$\mu_a$ Axion mass [eV]', fontsize=14)
    ax1.set_ylabel(r'$\Gamma_{lmn}^{-1}$ [s]', fontsize=14)
    #ax2.set_ylabel(r'$\Gamma_{lmn} r_g$', fontsize=14)

    # Invert the left y-axis (so longer timescales are higher)
    ax1.set_yscale('log')
    #ax2.set_yscale('log')
    ax1.invert_yaxis()

    # Set desired limits on left axis
    ax1.set_ylim(1e1, 1e-10)

    # Compute matching limits for the right axis
    year_to_ev_inv = 1 / 2.086e-23  # conversion from years to eV^-1
    y1_min, y1_max = ax1.get_ylim()
    y2_min = (2.086e-23 * r_g) / y1_max
    y2_max = (2.086e-23 * r_g) / y1_min
    #ax2.set_ylim(y2_max, y2_min)

    # --- Titles and labels ---
    ax1.set_title(r'Superradiance Growth Rate, $\Gamma_{nm\ell}$', 
                  fontsize=16)
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # --- Legends ---
    color_handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=4)
                     for i in range(len(l_values))]
    color_labels = [fr"$\ell$={l}" for l in l_values]
    legend1 = ax1.legend(color_handles, color_labels,
                         title=r'orbital ($\ell$)',
                         loc='upper right', frameon=True)

    style_handles = [Line2D([0], [0], color='black', lw=2,
                            linestyle=linestyles[i % len(linestyles)])
                     for i in range(len(spins))]
    style_labels = [f"a*={a_star:.3f}" for a_star in spins]
    legend2 = ax1.legend(style_handles, style_labels,
                         title=r'spin ($a^*$)',
                         loc='upper left', frameon=True)

    ax1.add_artist(legend1)
    ax1.add_artist(legend2)

    # --- Annotate BH mass on the plot ---
    ax1.text(0.15, 0.95,
        fr"$M_{{BH}} = {bh_mass:.1e}\ M_\odot$",
        transform=ax1.transAxes,
        fontsize=13,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none')
    )

    # --- Save the figure ---
    filename = f"SuperradianceRates_{int(bh_mass)}.png"
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"\n✅ Figure saved as '{filename}' in the current directory.\n")

    plt.show()



def main():
    blackholemass = 10  # [solar masses]
    data = compute_superradiance_data(blackholemass)
    plot_superradiance_data(data)


if __name__ == "__main__":
    main()
