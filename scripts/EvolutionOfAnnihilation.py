import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the ODE
def dN_dt(t, N, Gamma_sr, Gamma_a):
    return Gamma_sr * N - Gamma_a * N**2

# Simulation function
def simulate_N(Gamma_sr=1.0, Gamma_a=0.1, N0=1.0, t_span=(0, 50), num_points=500):
    # Time grid for output
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    # Solve ODE using Radau method
    sol = solve_ivp(
        dN_dt, 
        t_span, 
        [N0], 
        args=(Gamma_sr, Gamma_a), 
        method='Radau', 
        t_eval=t_eval
    )
    
    # Plot the result
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, sol.y[0], label=r"$N(t)$", linewidth=2)
    plt.title("Evolution of N over time", fontsize=14)
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Population N(t)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
    return sol

# Example run
simulate_N(Gamma_sr=1e4, Gamma_a=1e-79, N0=1.0, t_span=(0, int(1)), num_points=int(1e4))
