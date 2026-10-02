from __future__ import annotations

import csv
import json
import os
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_generator import simulate_shallow_water


def _ensure_eval_dir(pinn) -> str:
    out_dir = os.path.join(pinn.checkpoint_folder, "evaluation_plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _detrended_spectrum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trend = np.linspace(y[0], y[-1], len(y))
    y_detrended = y - trend
    n = len(y_detrended)
    dx = x[1] - x[0]
    yf = np.fft.fft(y_detrended)
    k = np.fft.fftfreq(n, d=dx) * 2 * np.pi
    pos = k > 0
    amp = np.abs(yf[pos]) / max(n, 1)
    return k[pos], amp


def _plot_profile_panel(ax, x, y_true, y_pred, ylabel: str, title: str) -> None:
    ax.plot(x, y_true, color="black", label="FOM")
    ax.plot(x, y_pred, "--", color="blue", label="PINN")
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True)
    ax.legend()


def _plot_spectrum_panel(ax, x, y_true, y_pred, quantity_name: str, title: str) -> None:
    k_true, amp_true = _detrended_spectrum(x, y_true)
    k_pred, amp_pred = _detrended_spectrum(x, y_pred)
    ax.semilogy(k_true, amp_true, marker='.', color='black', linestyle='none', label='FOM')
    ax.semilogy(k_pred, amp_pred, marker='.', color='blue', linestyle='none', label='PINN')
    ax.set_xlabel('Wavenumber k')
    # ax.set_ylabel('Amplitude')
    ax.set_ylabel(rf"$|\hat{{{quantity_name}_k}}|^2$")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True)
    ax.legend()


def shock_sampling_sanity_plot(
    pinn,
    regimes=None,
    times=None,
    n_per_time: int = 2000,
    save_name: str = "shock_sampling_sanity.png",
):
    """Save a histogram sanity check for shock-aware collocation sampling."""
    if regimes is None:
        regimes = [(float(r[0]), float(r[1]), str(r[2])) for r in getattr(pinn, "train_regimes", [])]
    if not regimes:
        raise RuntimeError("No regimes available. Call get_data() first.")
    if times is None:
        times = [0.1, 0.5, 1.0, float(getattr(pinn, "t_final", 2.5))]

    cfg = pinn.get_data_params.get("shock_aware_colloc", {}) if hasattr(pinn, "get_data_params") else {}
    w_excl = float(cfg.get("w_excl", 1.5))
    w_focus = float(cfg.get("w_focus", 6.0))
    focus_share = float(cfg.get("focus_share", 0.6))
    dry_hR_eps = float(cfg.get("dry_hR_eps", 0.5))
    Lx = float(getattr(pinn, "Lx", pinn.get_data_params.get("Lx", 100.0)))

    out_dir = os.path.join(pinn.checkpoint_folder, "sanity")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, save_name)

    ncols = len(times)
    nrows = len(regimes)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.8*nrows), constrained_layout=False)
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = np.array([axs])
    elif ncols == 1:
        axs = np.array([[a] for a in axs])

    for ri, reg in enumerate(regimes):
        hL, hR, shape = reg
        key = (float(hL), float(hR), str(shape))
        for ci, tt in enumerate(times):
            tt = float(tt)
            if float(hR) < dry_hR_eps:
                xS = None
                xs = np.random.uniform(0.0, Lx, (int(n_per_time), 1))
            else:
                xS = pinn._shock_x_at(key, tt)
                xs = pinn._sample_x_shock_aware(
                    xS=xS, Lx=Lx, n=int(n_per_time),
                    w_excl=w_excl, w_focus=w_focus, focus_share=focus_share,
                )
            ax = axs[ri, ci]
            ax.hist(xs[:, 0], bins=150, density=True, label="centers (collocation x)")
            if xS is not None:
                ax.axvspan(max(0.0, xS-w_excl), min(Lx, xS+w_excl), alpha=0.25)
                ax.axvline(xS, linestyle="--")
            ax.set_xlim(0.0, Lx)
            xs_str = "NA" if xS is None else f"{xS:.1f}"
            ax.set_title(f"(hL={hL}, hR={hR}) t={tt:.2f} xS={xs_str}")
            if ci == 0:
                ax.set_ylabel("density")
            if ri == nrows - 1:
                ax.set_xlabel("x")
            ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(f"Shock-aware sampling sanity (w_excl={w_excl}, w_focus={w_focus}, focus_share={focus_share})")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[sanity] Saved shock sampling sanity plot to {out_path}")
    return out_path


def evaluate_model(
    pinn,
    test_t: float = 1.0,
    test_h: tuple[float, float, str] = (12.0, 5.0, "perpendicular"),
    plot: bool = True,
) -> dict[str, Any]:
    """Evaluate the trained PINN on one regime/time and save diagnostic plots."""
    pinn.model.eval()
    test_t = float(test_t)
    test_h_left = float(test_h[0])
    test_h_right = float(test_h[1])
    test_shape = str(test_h[2])
    os.makedirs(pinn.checkpoint_folder + '/evaluation_plots', exist_ok=True)

    start_time_num = time.time()
    Lx = float(pinn.get_data_params.get("Lx", 100.0))
    Nx = int(pinn.get_data_params.get("Nx", 402))
    dt = float(pinn.get_data_params.get("dt", 1e-4))
    test_snapshots = simulate_shallow_water(
        h_left=test_h_left,
        h_right=test_h_right,
        dam_shape=test_shape,
        Lx=Lx,
        Nx=Nx,
        t_final=test_t,
        dt=dt,
        sample_interval=test_t,
        plot=False,
    )
    elapsed_time_num = time.time() - start_time_num
    actual_solution = test_snapshots[-1]
    x_actual = actual_solution['x']
    h_actual = actual_solution['h']
    u_actual = actual_solution['u']
    q_actual = pinn._q_from_hu_np(h_actual, u_actual)

    start_time_pinn = time.time()
    pred_fields = pinn._predict_fields_on_grid(
        x=x_actual,
        t=test_t,
        h_left=test_h_left,
        h_right=test_h_right,
    )
    elapsed_time_pinn = time.time() - start_time_pinn

    h_pred_test = pred_fields["h_pred"]
    q_pred_test = pred_fields["q_pred"]

    error_h = np.linalg.norm(h_pred_test - h_actual, 2) / max(np.linalg.norm(h_actual, 2), 1e-12)
    error_q = np.linalg.norm(q_pred_test - q_actual, 2) / max(np.linalg.norm(q_actual, 2), 1e-12)

    names = ['h', 'q']
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=False)
    fig.suptitle(
        f'Numerical Solution (Rusanov) time cost = {elapsed_time_num:.4f}s vs. '
        f'PINN time cost = {elapsed_time_pinn:.4f}s'
    )
    for i, y in enumerate(zip([h_actual, q_actual], [h_pred_test, q_pred_test])):
        k_pos, amp = _detrended_spectrum(x_actual, y[0])
        k_pred_pos, amp_pred = _detrended_spectrum(x_actual, y[1])

        if i == 0:
            ax[0, i].plot(x_actual, h_actual, 'k-', label='Actual h')
            ax[0, i].plot(x_actual, h_pred_test, 'b--', label=f'Predicted h, relative_L2={error_h:.2%}')
            ax[0, i].set_ylabel('Water Depth h (m)')
            ax[0, i].set_title(f'Water Depth at t = {test_t:.2f}s, h_left = {test_h_left}m, h_right={test_h_right}m')
        else:
            ax[0, i].plot(x_actual, q_actual, 'k-', label='Actual q')
            ax[0, i].plot(x_actual, q_pred_test, 'b--', label=f'Predicted q, relative_L2={error_q:.2%}')
            ax[0, i].set_ylabel('Discharge q (m^2/s)')
            ax[0, i].set_title(f'Discharge at t = {test_t:.2f} s, h_left = {test_h_left}m, h_right={test_h_right}m')
        ax[0, i].set_xlabel('x (m)')
        ax[0, i].legend()
        ax[0, i].grid(True)

        title = f"{names[i]} - trend at t={test_t:.2f}s"
        ax[1, i].semilogy(k_pos, amp, marker='.', color='black', linestyle='none', label=f'Actual {names[i]}')
        ax[1, i].semilogy(k_pred_pos, amp_pred, color='blue', marker='.', linestyle='none', label=f'Predicted {names[i]}')
        ax[1, i].set_xlabel("Wavenumber k")
        # ax[1, i].set_ylabel("Amplitude")
        ax[1, i].set_ylabel(rf"$|\hat{{{names[i]}_k}}|^2$")
        ax[1, i].set_title(title)
        ax[1, i].grid(True)
        ax[1, i].legend()

    out_path = f'./{pinn.checkpoint_folder}/evaluation_plots/test_hleft={test_h_left}m_hright={test_h_right}m_t={test_t}s.png'
    try:
        fig.savefig(out_path)
    except RecursionError as e:
        print(f"[warn] savefig recursion error (evaluation plot) -> skipping: {e}")
    if plot and getattr(pinn, "display", False):
        plt.show()
    plt.close('all')

    return {
        "t": test_t,
        "h_left": test_h_left,
        "h_right": test_h_right,
        "error_h": float(error_h),
        "error_q": float(error_q),
    }


# def plot_trom_style_final_profiles(
#     pinn,
#     test_h: list[tuple[float, float, str]],
#     test_t: float | list[float] = 2.5,
#     field: str = "h",
#     plot_kind: str = "profile",
#     Lx: float | None = None,
#     Nx: int | None = None,
#     dt: float | None = None,
#     save_name: str | None = None,
# ) -> str:
#     """Create tROM-style multi-panel figures for final-time profiles or spectra."""
#     field = str(field).lower()
#     plot_kind = str(plot_kind).lower()
#     if field not in {"h", "q"}:
#         raise ValueError(f"field must be 'h' or 'q', got: {field}")
#     if plot_kind not in {"profile", "spectrum"}:
#         raise ValueError(f"plot_kind must be 'profile' or 'spectrum', got: {plot_kind}")

#     if isinstance(test_t, (list, tuple, np.ndarray)):
#         test_times = [float(tt) for tt in test_t]
#     else:
#         test_times = [float(test_t)]

#     if len(test_times) not in {1, 2}:
#         raise ValueError(f"test_t must be a scalar or a list/tuple of exactly 2 times, got {test_t}")
#     n_times = len(test_times)
#     if n_times == 1 and len(test_h) != 6:
#         raise ValueError(f"For scalar test_t, expected exactly 6 parameter cases to mirror the tROM layout, got {len(test_h)}")
#     if n_times == 2 and len(test_h) != 3:
#         raise ValueError(f"For 2-element test_t, expected exactly 3 parameter cases (one per row of the 3x2 layout), got {len(test_h)}")

#     os.makedirs(pinn.checkpoint_folder + '/evaluation_plots', exist_ok=True)
#     Lx = float(pinn.get_data_params.get("Lx", 100.0) if Lx is None else Lx)
#     Nx = int(pinn.get_data_params.get("Nx", 402) if Nx is None else Nx)
#     dt = float(pinn.get_data_params.get("dt", 1e-4) if dt is None else dt)

#     def _extract_plot_payload(fin: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
#         x = fin["x"]
#         if field == "h":
#             y_true = fin["h_true"]
#             y_pred = fin["h_pred"]
#             ylabel = "h"
#             quantity_name = "Water depth"
#         else:
#             y_true = fin["q_true"]
#             y_pred = fin["q_pred"]
#             ylabel = "q"
#             quantity_name = "Discharge"
#         return x, y_true, y_pred, ylabel, quantity_name

#     if len(test_times) == 1:
#         fig, axs = plt.subplots(3, 2, figsize=(13, 15), constrained_layout=False)
#         axs = axs.flatten(order='C')

#         for i, regime in enumerate(test_h):
#             h_left, h_right, dam_shape = float(regime[0]), float(regime[1]), str(regime[2])
#             t_plot = test_times[0]
#             data = pinn._relative_error_time_series(
#                 h_left=h_left, h_right=h_right, dam_shape=dam_shape,
#                 Lx=Lx, Nx=Nx, t_final=t_plot, dt=dt,
#                 sample_interval=t_plot, field=field,
#             )
#             fin = data["final"]
#             x, y_true, y_pred, ylabel, quantity_name = _extract_plot_payload(fin)

#             if plot_kind == "profile":
#                 if field == "h":
#                     title = f"Water depth profiles ( h_L = {h_left}, h_R = {h_right}, t = {t_plot:.1f} )"
#                 else:
#                     title = f"Discharge Profile ( h_L = {h_left}, h_R = {h_right}, t = {t_plot:.1f} )"
#                 _plot_profile_panel(axs[i], x, y_true, y_pred, ylabel, title)
#             else:
#                 title = f"{quantity_name} spectrum ( h_L = {h_left}, h_R = {h_right}, t = {t_plot:.1f} )"
#                 _plot_spectrum_panel(axs[i], x, y_true, y_pred, quantity_name, title)
#     else:
#         fig, axs = plt.subplots(3, 2, figsize=(13, 15), constrained_layout=False)
#         axs = np.asarray(axs)
#         if axs.ndim == 1:
#             axs = axs[:, None]

#         for i, regime in enumerate(test_h):
#             h_left, h_right, dam_shape = float(regime[0]), float(regime[1]), str(regime[2])
#             for j, t_plot in enumerate(test_times):
#                 data = pinn._relative_error_time_series(
#                     h_left=h_left, h_right=h_right, dam_shape=dam_shape,
#                     Lx=Lx, Nx=Nx, t_final=t_plot, dt=dt,
#                     sample_interval=t_plot, field=field,
#                 )
#                 fin = data["final"]
#                 x, y_true, y_pred, ylabel, quantity_name = _extract_plot_payload(fin)

#                 if plot_kind == "profile":
#                     if field == "h":
#                         title = f"Water depth profiles ( h_L = {h_left}, h_R = {h_right}, t = {t_plot:.1f} )"
#                     else:
#                         title = f"Discharge Profile ( h_L = {h_left}, h_R = {h_right}, t = {t_plot:.1f} )"
#                     _plot_profile_panel(axs[i, j], x, y_true, y_pred, ylabel, title)
#                 else:
#                     title = f"{quantity_name} spectrum ( h_L = {h_left}, h_R = {h_right}, t = {t_plot:.1f} )"
#                     _plot_spectrum_panel(axs[i, j], x, y_true, y_pred, quantity_name, title)

#     plt.tight_layout()
#     if save_name is None:
#         base = f"trom_style_final_{plot_kind}_{field}_{test_h}"
#         if len(test_times) == 1:
#             save_name = f"{base}.png"
#         else:
#             t_tag = "_".join(f"{tt:g}" for tt in test_times)
#             save_name = f"{base}_t_{t_tag}.png"
#     out_path = os.path.join(pinn.checkpoint_folder, 'evaluation_plots', save_name)
#     fig.savefig(out_path, dpi=200, bbox_inches='tight')
#     plt.close(fig)
#     return out_path

def plot_trom_style_final_profiles(
    pinn,
    test_h: list[tuple[float, float, str]],
    test_t: float | list[float] = 2.5,
    field: str = "h",
    plot_kind: str = "profile",
    Lx: float | None = None,
    Nx: int | None = None,
    dt: float | None = None,
    save_name: str | None = None,
) -> list[str]:
    """Create and save one figure per regime and time."""
    field = str(field).lower()
    plot_kind = str(plot_kind).lower()

    if field not in {"h", "q"}:
        raise ValueError(f"field must be 'h' or 'q', got: {field}")
    if plot_kind not in {"profile", "spectrum"}:
        raise ValueError(
            f"plot_kind must be 'profile' or 'spectrum', got: {plot_kind}"
        )

    if isinstance(test_t, (list, tuple, np.ndarray)):
        test_times = [float(tt) for tt in test_t]
    else:
        test_times = [float(test_t)]

    if not test_times:
        raise ValueError("test_t must contain at least one time")
    if not test_h:
        raise ValueError("test_h must contain at least one regime")

    out_dir = _ensure_eval_dir(pinn)
    Lx = float(pinn.get_data_params.get("Lx", 100.0) if Lx is None else Lx)
    Nx = int(pinn.get_data_params.get("Nx", 402) if Nx is None else Nx)
    dt = float(pinn.get_data_params.get("dt", 1e-4) if dt is None else dt)

    out_paths = []

    for regime in test_h:
        h_left = float(regime[0])
        h_right = float(regime[1])
        dam_shape = str(regime[2])

        for t_plot in test_times:
            data = pinn._relative_error_time_series(
                h_left=h_left,
                h_right=h_right,
                dam_shape=dam_shape,
                Lx=Lx,
                Nx=Nx,
                t_final=t_plot,
                dt=dt,
                sample_interval=t_plot,
                field=field,
            )
            fin = data["final"]
            x = fin["x"]

            if field == "h":
                y_true = fin["h_true"]
                y_pred = fin["h_pred"]
                ylabel = "h"
                quantity_name = "h"
            else:
                y_true = fin["q_true"]
                y_pred = fin["q_pred"]
                ylabel = "q"
                quantity_name = "q"

            fig, ax = plt.subplots(
                figsize=(7, 5),
                constrained_layout=False,
            )

            title = (
                f"PINN t={t_plot:g} "
                rf"$\mathbf{{(h_{{L}}={h_left:g},\ h_{{R}}={h_right:g})}}$"
            )

            if plot_kind == "profile":
                _plot_profile_panel(
                    ax, x, y_true, y_pred, ylabel, title
                )
            else:
                _plot_spectrum_panel(
                    ax, x, y_true, y_pred, quantity_name, title
                )

            fig.tight_layout()

            if save_name is None:
                case_name = (
                    f"trom_style_final_{plot_kind}_{field}_"
                    f"hL_{h_left:g}_hR_{h_right:g}_"
                    f"t_{t_plot:g}.png"
                )
            else:
                stem, extension = os.path.splitext(save_name)
                case_name = (
                    f"{stem}_hL_{h_left:g}_hR_{h_right:g}_"
                    f"t_{t_plot:g}{extension or '.png'}"
                )

            out_path = os.path.join(out_dir, case_name)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            out_paths.append(out_path)

    return out_paths


def plot_trom_style_error_evolution(
    pinn,
    test_h: list[tuple[float, float, str]],
    field: str = "h",
    Lx: float | None = None,
    Nx: int | None = None,
    t_final: float | None = None,
    dt: float | None = None,
    sample_interval: float = 0.01,
    save_name: str | None = None,
    extra_table_h: list[tuple[float, float, str]] | None = None,
) -> str:
    """Create a 2x3 multi-panel relative-error-vs-time figure."""
    field = str(field).lower()
    if field not in {"h", "q", "state"}:
        raise ValueError(f"field must be 'h', 'q', or 'state', got: {field}")

    out_dir = _ensure_eval_dir(pinn)
    Lx = float(pinn.get_data_params.get("Lx", 100.0) if Lx is None else Lx)
    Nx = int(pinn.get_data_params.get("Nx", 402) if Nx is None else Nx)
    t_final = float(pinn.get_data_params.get("t_final", 2.5) if t_final is None else t_final)
    dt = float(pinn.get_data_params.get("dt", 1e-4) if dt is None else dt)

    n_cases = len(test_h)
    if n_cases == 0:
        raise ValueError("test_h must contain at least one parameter case")
    ncols = 2 if n_cases > 1 else 1
    nrows = int(np.ceil(n_cases / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows), constrained_layout=False, squeeze=False)
    axs = axs.flatten(order="C")
    for ax in axs[n_cases:]:
        ax.set_visible(False)

    if field == "h":
        title_stub = "Relative Error Evolution in h"
        ylabel = "Relative Error in h"
    elif field == "q":
        title_stub = "Relative Error Evolution in q"
        ylabel = "Relative Error in q"
    else:
        title_stub = "Relative Error Evolution in state"
        ylabel = "Relative Error in state"

    error_table = []
    for i, regime in enumerate(test_h):
        h_left, h_right, dam_shape = float(regime[0]), float(regime[1]), str(regime[2])
        data = pinn._relative_error_time_series(
            h_left=h_left,
            h_right=h_right,
            dam_shape=dam_shape,
            Lx=Lx,
            Nx=Nx,
            t_final=t_final,
            dt=dt,
            sample_interval=sample_interval,
            field=field,
        )
        ax = axs[i]
        mask = np.isfinite(data["rel_error"])
        ax.semilogy(data["t"][mask], data["rel_error"][mask], color="blue", label=f"PINN error in {field}")
        ax.set_title(
            f"{title_stub} "
            rf"$\mathbf{{(h_{{L}}={h_left:g},\ h_{{R}}={h_right:g})}}$",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

        finite_t = np.asarray(data["t"])[mask]
        finite_error = np.asarray(data["rel_error"])[mask]
        error_table.append({
            "h_L": h_left,
            "h_R": h_right,
            "dam_shape": dam_shape,
            "field": field,
            "relative_error_t_1": float(
                np.interp(1.0, finite_t, finite_error)
            ),
            "relative_error_t_2": float(
                np.interp(2.0, finite_t, finite_error)
            ),
        })

    existing_regimes = {
        (float(regime[0]), float(regime[1]), str(regime[2]))
        for regime in test_h
    }

    for regime in extra_table_h or []:
        h_left = float(regime[0])
        h_right = float(regime[1])
        dam_shape = str(regime[2])
        regime_key = (h_left, h_right, dam_shape)

        if regime_key in existing_regimes:
            continue

        data = pinn._relative_error_time_series(
            h_left=h_left,
            h_right=h_right,
            dam_shape=dam_shape,
            Lx=Lx,
            Nx=Nx,
            t_final=t_final,
            dt=dt,
            sample_interval=sample_interval,
            field=field,
        )

        mask = np.isfinite(data["rel_error"])
        finite_t = np.asarray(data["t"])[mask]
        finite_error = np.asarray(data["rel_error"])[mask]

        error_table.append({
            "h_L": h_left,
            "h_R": h_right,
            "dam_shape": dam_shape,
            "field": field,
            "relative_error_t_1": float(
                np.interp(1.0, finite_t, finite_error)
            ),
            "relative_error_t_2": float(
                np.interp(2.0, finite_t, finite_error)
            ),
        })

    plt.tight_layout()
    if save_name is None:
        save_name = f"trom_style_error_evolution_{field}.png"
    out_path = os.path.join(out_dir, save_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    table_path = os.path.join(
        out_dir,
        f"trom_style_error_values_{field}_t_1_t_2.csv",
    )
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=error_table[0].keys(),
        )
        writer.writeheader()
        writer.writerows(error_table)
    return out_path


def test_model(
    pinn,
    test_t: list[float] = [0.5, 1.0, 1.5, 2.0],
    test_h: list[tuple[float, float, str]] = [
        (3.0, 0.0, "perpendicular"), (5.0, 0.0, "perpendicular"),
        (7.0, 0.0, "perpendicular"), (10.0, 0.0, "perpendicular"),
        (12.0, 0.0, "perpendicular"), (15.0, 0.0, "perpendicular"),
        (17.0, 0.0, "perpendicular"),
    ],
    plot: bool = False,
    paper_eval_kwargs: dict | None = None,
) -> None:
    """Run evaluation across multiple times/regimes and save a JSON summary."""
    results = []
    for t in test_t:
        for regime in test_h:
            res = evaluate_model(pinn, test_t=t, test_h=regime, plot=plot)
            results.append(res)

    out_path = os.path.join(pinn.checkpoint_folder, "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test results saved to {out_path}")

    if paper_eval_kwargs is not None:
        pinn.paper_style_eval(**paper_eval_kwargs)

    paper_regimes = [
        (12.0, 0.0, "perpendicular"),
        (12.0, 7.0, "perpendicular"),
        (15.0, 4.0, "perpendicular"),
        (18.0, 0.0, "perpendicular"),
        (26.0, 0.14, "perpendicular"),
        (26.0, 7.0, "perpendicular"),
    ]

    # plot_trom_style_final_profiles(pinn, test_h=paper_regimes, test_t=2.0, field="h")
    # plot_trom_style_final_profiles(pinn, test_h=paper_regimes, plot_kind="spectrum", test_t=2.0, field="h")
    plot_trom_style_error_evolution(pinn, test_h=paper_regimes, field="h", sample_interval=0.01, extra_table_h=test_h)
    # plot_trom_style_final_profiles(pinn, test_h=paper_regimes, test_t=2.0, field="q")
    # plot_trom_style_final_profiles(pinn, test_h=paper_regimes, plot_kind="spectrum", test_t=2.0, field="q")
    plot_trom_style_error_evolution(pinn, test_h=paper_regimes, field="q", sample_interval=0.01, extra_table_h=test_h)

    # extr_regimes = [
    #     (9.0, 4.0, "perpendicular"),
    #     (20.0, 8.5, "perpendicular"),
    #     (29.0, 4.0, "perpendicular"),
    #     (32.0, 4.0, "perpendicular"),
    #     (29.0, 8.5, "perpendicular"),
    #     (32.0, 10.0, "perpendicular"),
    # ]
    # plot_trom_style_final_profiles(pinn, test_h=extr_regimes, plot_kind="profile", test_t=2.5, field="h")
    # plot_trom_style_final_profiles(pinn, test_h=extr_regimes, plot_kind="spectrum", test_t=2.5, field="h")
    # plot_trom_style_final_profiles(pinn, test_h=extr_regimes, plot_kind="profile", test_t=2.5, field="q")
    # plot_trom_style_final_profiles(pinn, test_h=extr_regimes, plot_kind="spectrum", test_t=2.5, field="q")

    # extr_regimes1 = [
    #     (9.0, 4.0, "perpendicular"),
    #     (20.0, 8.5, "perpendicular"),
    #     (32.0, 10.0, "perpendicular"),
    # ]
    # plot_trom_style_final_profiles(pinn, test_h=extr_regimes1, plot_kind="profile", test_t=[3.0, 4.0], field="h")
    # plot_trom_style_final_profiles(pinn, test_h=extr_regimes1, plot_kind="spectrum", test_t=[3.0, 4.0], field="h")
    # plot_trom_style_final_profiles(pinn, test_h=extr_regimes1, plot_kind="profile", test_t=[3.0, 4.0], field="q")
    # plot_trom_style_final_profiles(pinn, test_h=extr_regimes1, plot_kind="spectrum", test_t=[3.0, 4.0], field="q")

    for field in ("h", "q"):
        for plot_kind in ("profile", "spectrum"):
            plot_trom_style_final_profiles(
                pinn,
                test_h=test_h,
                test_t=test_t,
                field=field,
                plot_kind=plot_kind,
                save_name=f"requested_{plot_kind}_{field}.png",
            )