import matplotlib.pyplot as plt
import numpy as np
import time
import os

from typing import Any

def simulate_shallow_water(
    h_left: float = 10.0,
    h_right: float = 0.0,
    dam_shape: str = 'perpendicular',
    Lx: float = 100.0,
    Nx: int = 1002,
    t_final: float = 2.0,
    dt: float = 1e-3,
    sample_interval: float = 0.1,
    plot: bool = False
) -> list[dict[str, Any]]:
    """
    Run a 1D shallow water dam-break simulation and collect snapshots.

    Parameters
    ----------
    h_left : float, optional
        Upstream water depth (default is 10.0 m).
    h_right : float, optional
        Downstream water depth (dry bed, default is 0.0 m).
    dam_shape : str, optional
        Shape of the dam; only 'perpendicular' is implemented (default).
    Lx : float, optional
        Total domain length in the x-direction (default is 100.0 m).
    Nx : int, optional
        Number of grid points including ghost cells (default is 1002).
    t_final : float, optional
        Final simulation time in seconds (default is 2.0 s).
    dt : float, optional
        Time step for the Euler update (default is 1e-3 s).
    sample_interval : float, optional
        Time interval between saving snapshots (default is 0.1 s).
    plot : bool, optional
        If True, display (and save) inline plots of each snapshot (default is False).

    Returns
    -------
    snapshots : list of dict
        A list of snapshot dictionaries with keys:
          - 'x': np.ndarray of interior spatial points, shape (Nx-2,)
          - 't': float, simulation time at snapshot
          - 'h': np.ndarray of water depths at 'x', shape (Nx-2,)
          - 'u': np.ndarray of velocity u at 'x', shape (Nx-2,)
          - 'param': np.ndarray of [h_left, h_right], shape (2,)

    Raises
    ------
    ValueError
        If `dam_shape` is not 'perpendicular'.
    """
    g = 9.81 # Acceleration due to gravity (m/s^2)
    dx = Lx / (Nx - 2) # Grid spacing in x-direction (excluding ghost cells)
    x = np.linspace(-dx/2, Lx + dx/2, Nx)  # computational grid (including ghost cells)
    x0 = Lx / 2.0 # Position of the dam

    # Set initial condition
    if dam_shape == 'perpendicular':
        h = np.where(x < x0, h_left, h_right)
    # elif dam_shape == 'inclined':
    #     # Inclined dam: h varies linearly from h_left at x_start to h_right at x_end
    #     x_start = x0 - Lx / 10  # Start of the inclined dam
    #     x_end = x0 + Lx / 10    # End of the inclined dam
    #     h = np.where(x < x_start, h_left,
    #                  np.where(x > x_end, h_right,
    #                           h_left - ((h_left - h_right)/(x_end - x_start)) * (x - x_start)))
    # elif dam_shape == 'parabolic':
    #     # Parabolic dam: h varies parabolically from h_left at x_start to h_right at x_end
    #     x_start = x0 - Lx / 10  # Start of the parabolic dam
    #     x_end = x0 + Lx / 10    # End of the parabolic dam
    #     a = (h_right - h_left) / (x_end - x_start)**2
    #     h = np.where(x < x_start, h_left,
    #                  np.where(x > x_end, h_right,
    #                           h_left + a * (x - x_start)**2))
    else:
        raise ValueError("Only 'perpendicular' dam is implemented in this simulation.")

    def apply_boundary_conditions(U):
        """
        Apply zero-gradient (outflow/Neumann) boundary conditions:
        copy first interior cell into the ghost cell on each side.
        """
        U[0, :] = U[1, :]
        U[-1, :] = U[-2, :]
        return U

    def compute_fluxes(U):
        """
        Compute fluxes in x-direction and primitive variables.
        """
        h = U[:, 0]
        hu = U[:, 1]
        h_safe = np.where(h < 1e-6, 1e-6, h) # Apply minimum depth threshold to avoid division by zero
        u = hu / h_safe # Primitive variables
        c = np.sqrt(g * h_safe) # Wave speeds
        F = np.zeros_like(U) # Fluxes in x-direction
        F[:, 0] = hu
        F[:, 1] = hu * u + 0.5 * g * h_safe**2
        return F, u, c

    def compute_rhs(U):
        """
        Compute the right-hand side (RHS) of the shallow water equations in 1D.
        """
        U = apply_boundary_conditions(U)
        F, u, c = compute_fluxes(U)
        lambda_x = np.abs(u) + c
        smax_x = np.maximum(lambda_x[:-1], lambda_x[1:]) # Maximum wave speeds
        delta_U_x = U[1:, :] - U[:-1, :]
        F_half_x = 0.5 * (F[1:, :] + F[:-1, :]) - 0.5 * smax_x[:, None] * delta_U_x # Numerical fluxes (Rusanov flux)
        RHS = np.zeros_like(U)
        RHS[1:-1, :] = - (F_half_x[1:, :] - F_half_x[:-1, :]) / dx # Divergence of fluxes
        return RHS
        
    u = np.zeros_like(h)  # Zero initial velocity in x-direction
    U = np.zeros((Nx, 2))
    # Conserved variables: U = [h, h*u]
    U[:, 0] = h
    U[:, 1] = h * u

    snapshots = []
    t = 0.0

    #Adding snapshot at initial t=0
    h_numerical = U[1:-1, 0]
    hu_numerical = U[1:-1, 1]
    h_safe = np.where(h_numerical < 1e-6, 1e-6, h_numerical)
    u_numerical = hu_numerical / h_safe
    snapshots.append({
        'x': x[1:-1].copy(),   # interior grid points
        't': t,
        'h': h_numerical.copy(),
        'u': u_numerical.copy(),
        'param': np.array([h_left, h_right])
    })
    
    if plot:
        from IPython.display import display
        os.makedirs('./snapshot_plots', exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        h_plot, = axs[0].plot(x[1:-1], h_numerical, label="Numerical h", color='black')
        axs[0].set_ylabel("h (m)", fontsize=12)
        axs[0].legend(fontsize=12)
        axs[0].grid(True)
        u_plot, = axs[1].plot(x[1:-1], u_numerical, label="Numerical u", color='black')
        axs[1].set_xlabel("x (m)", fontsize=12)
        axs[1].set_ylabel("u (m/s)", fontsize=12)
        axs[1].legend(fontsize=12)
        axs[1].grid(True)
        fig.suptitle(f"Dam Break at t = {t:.2f} s ({dam_shape.capitalize()} Dam)", fontsize=14)
        plt.tight_layout()
        fig.savefig(f'./experiments/snapshot_plots/hl={h_left}_hr={h_right}_t={t:0.1f}s_snap.png')
        display_handle = display(fig, display_id=True)
        time.sleep(0.4)

    next_sample_time = sample_interval
    while t < t_final - 1e-8:
        # Simple forward Euler update
        RHS = compute_rhs(U)
        U = U + dt * RHS
        t += dt
        
        if t >= next_sample_time - 1e-8:
            # extract interior (exclude ghost cells)
            h_numerical = U[1:-1, 0]
            hu_numerical = U[1:-1, 1]
            h_safe = np.where(h_numerical < 1e-6, 1e-6, h_numerical)
            u_numerical = hu_numerical / h_safe
            snapshots.append({
                'x': x[1:-1].copy(),   # interior grid points
                't': t,
                'h': h_numerical.copy(),
                'u': u_numerical.copy(),
                'param': np.array([h_left, h_right])
            })

            next_sample_time += sample_interval
            if plot:
                h_plot.set_data(x[1:-1], h_numerical)
                u_plot.set_data(x[1:-1], u_numerical)
                for ax in axs:
                    ax.relim()
                    ax.autoscale_view(True, True)
                fig.suptitle(f"Dam Break at t = {t:.2f} s ({dam_shape.capitalize()} Dam)", fontsize=14)
                display_handle.update(fig)
    
                time.sleep(0.4) #don't use plt.pause, it creates second figure
                fig.savefig(f'./experiments/snapshot_plots/hl={h_left}_hr={h_right}_t={t:.1f}s_snap.png')
                
    plt.close('all')

    return snapshots