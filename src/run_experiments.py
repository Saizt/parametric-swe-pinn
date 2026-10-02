import sys, os, json, argparse, shutil, glob, csv, time
from datetime import datetime
import torch, torch.nn as nn, torch.optim as optim
import numpy as np

import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt

from data_generator import simulate_shallow_water

from pinn import Linear_PINN
from experiment_utils import TimingRecorder, set_global_seed, save_run_manifest, snapshot_sources, dump_json
from evaluation_pipeline import run_evaluation_plan

# record the directory where run_experiments.py lives
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# helper to parse “h_left,h_right,shape”
def parse_regime(s: str):
    hL, hR, shape = s.split(',')
    return (float(hL), float(hR), shape)


def _deep_update(d: dict, u: dict) -> dict:
    """Recursively update dict d with values from u (in-place) and return d."""
    for k, v in (u or {}).items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _ns_copy(ns: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(ns))


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _find_latest_experiment_folder(root_dir: str) -> str:
    """Find latest experiments/<timestamp> folder under root_dir (best-effort)."""
    exp_root = os.path.join(root_dir, "experiments")
    if not os.path.isdir(exp_root):
        return ""
    cands = [os.path.join(exp_root, d) for d in os.listdir(exp_root)]
    cands = [d for d in cands if os.path.isdir(d)]
    if not cands:
        return ""
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def _load_metrics_from_folder(exp_folder: str) -> dict:
    """Load a small set of key metrics from *_summary.json files saved by paper_style_eval.

    For greedy sweeps we keep the CSV compact:
      - paper_h_avg := h_E_L2(L2)_avg_mean
      - paper_q_avg := q_E_L2(L2)_avg_mean
      - paper_score := 0.5*(paper_h_avg + paper_q_avg)

    Returns a flat dict of numeric metrics when possible.
    """
    out = {}
    if not exp_folder or not os.path.isdir(exp_folder):
        return out

    # Prefer explicit paper summary, but accept anything ending with _summary.json
    patterns = [
        # New location: experiments/<ts>/paper_style_eval/<files>
        os.path.join(exp_folder, "paper_style_eval", "*summary.json"),
        os.path.join(exp_folder, "paper_style_eval", "*paper*_summary.json"),
        # Backward-compatible fallbacks
        os.path.join(exp_folder, "*paper*_summary.json"),
        os.path.join(exp_folder, "*summary.json"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    # De-dup while preserving order
    seen = set()
    uniq_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq_files.append(f)

    # Target keys we care about (paper-style MC/grid summary json)
    H_KEY = "h_E_L2(L2)_avg_mean"
    Q_KEY = "q_E_L2(L2)_avg_mean"

    metric_files = []
    h_val = None
    q_val = None

    # Load candidates; first file that contains the target keys wins
    for f in uniq_files:
        try:
            with open(f, "r") as fh:
                js = json.load(fh)
            metric_files.append(os.path.basename(f))

            # Some summaries are flat dicts; be defensive anyway
            if isinstance(js, dict):
                if h_val is None and H_KEY in js and isinstance(js[H_KEY], (int, float)):
                    h_val = float(js[H_KEY])
                if q_val is None and Q_KEY in js and isinstance(js[Q_KEY], (int, float)):
                    q_val = float(js[Q_KEY])

            # If we found both, stop early
            if (h_val is not None) and (q_val is not None):
                break
        except Exception:
            continue

    if metric_files:
        out["metric_files"] = ",".join(metric_files)

    if h_val is not None:
        out["paper_h_avg"] = h_val
    if q_val is not None:
        out["paper_q_avg"] = q_val

    if (h_val is not None) and (q_val is not None):
        out["paper_score_0p5"] = 0.5 * (h_val + q_val)

    return out


def _write_greedy_csv(rows: list, out_csv_path: str):
    if not rows:
        return
    # union of keys
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    _safe_mkdir(os.path.dirname(out_csv_path))
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _apply_eval_paper_figures_overrides(ns: argparse.Namespace) -> None:
    """Apply eval_paper_figures overrides (currently: replace test_h with ROM-paper pairs).

    NOTE: In greedy_search mode we must re-apply this AFTER JSON overrides, because
    eval_paper_figures may be enabled via the greedy baseline/experiment overrides.
    """
    if getattr(ns, "eval_paper_figures", False):
        ns.test_h = [
            (12.0, 0.0, "perpendicular"),
            (12.0, 7.0, "perpendicular"),
            (15.0, 4.0, "perpendicular"),
            (18.0, 0.0, "perpendicular"),
            (26.0, 0.14, "perpendicular"),
            (26.0, 7.0, "perpendicular"),
        ]

def _resolve_path(pth: str, root_cwd: str) -> str:
    if pth is None:
        return None
    pth = str(pth)
    if os.path.isabs(pth):
        return pth
    return os.path.normpath(os.path.join(root_cwd, pth))

def _apply_train_grid_json(ns: argparse.Namespace, root_cwd: str) -> None:
    """Apply train_grid_json by expanding train_regime and overriding matching args keys.

    Important for greedy_search: train_grid_json may be introduced via JSON overrides.
    """
    pth = getattr(ns, "train_grid_json", None)
    if not pth:
        return
    pth = _resolve_path(pth, root_cwd)
    if not pth or not os.path.exists(pth):
        raise FileNotFoundError(f"train_grid_json not found: {pth}")
    with open(pth, "r") as f:
        train_grid = json.load(f)
    shape = train_grid.get("shape", "perpendicular")
    hL_list = [float(x) for x in train_grid["h_L"]]
    hR_list = [float(x) for x in train_grid["h_R"]]
    ns.train_regime = [(hL, hR, shape) for hL in hL_list for hR in hR_list]

    opts = vars(ns)
    for k, v in train_grid.items():
        if k in opts:
            setattr(ns, k, v)

def _apply_paper_eval_grid_json(ns: argparse.Namespace, root_cwd: str) -> None:
    """Apply paper_eval_grid_json by expanding paper_eval_grid_hL/paper_eval_grid_hR and overriding matching args keys.

    Important for greedy_search: paper_eval_grid_json may be introduced via JSON overrides.
    """
    pth = getattr(ns, "paper_eval_grid_json", None)
    if not pth:
        return
    pth = _resolve_path(pth, root_cwd)
    if not pth or not os.path.exists(pth):
        raise FileNotFoundError(f"paper_eval_grid_json not found: {pth}")
    with open(pth, "r") as f:
        eval_grid = json.load(f)
    ns.paper_eval_grid_hL = [float(x) for x in eval_grid["h_L"]]
    ns.paper_eval_grid_hR = [float(x) for x in eval_grid["h_R"]]

    opts = vars(ns)
    for k, v in eval_grid.items():
        if k in opts:
            setattr(ns, k, v)

def _estimate_shock_x_from_truth_snapshot(x: np.ndarray, y_in: np.ndarray, indicator: str = "h",
                                          smooth_window: int = 7, boundary_buf_frac: float = 0.02) -> float:
    """Robust shock locator used for sanity overlay.

    - Light moving-average smoothing to reduce peak-jumps.
    - Ignore a small boundary buffer.
    - For indicator='h' prefer the strongest negative gradient (compression shock).
    """
    if x.ndim != 1 or y_in.ndim != 1 or len(x) != len(y_in) or len(x) < 7:
        return float(x[len(x)//2])

    y = y_in

    w = int(max(1, smooth_window))
    if w >= 3:
        if (w % 2) == 0:
            w += 1
        kernel = np.ones(w, dtype=float) / float(w)
        y = np.convolve(y, kernel, mode='same')

    dx = float(x[1] - x[0])
    g = np.zeros_like(y)
    g[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
    g[0] = (y[1] - y[0]) / dx
    g[-1] = (y[-1] - y[-2]) / dx

    buf = max(2, int(boundary_buf_frac * len(x)))
    i0 = buf
    i1 = len(x) - buf
    if i1 <= i0 + 2:
        i0 = 2
        i1 = len(x) - 2

    gg = g[i0:i1]
    if indicator.lower() == 'h':
        j_local = int(np.argmin(gg))
    else:
        j_local = int(np.argmax(np.abs(gg)))
    j = j_local + i0
    return float(x[j])

def _shock_exclusion_overlay(pinn_obj, regimes, times, save_name: str,
                            Lx: float, Nx: int, dt: float, t_final: float,
                            w_excl: float, indicator: str = "h", dry_hR_eps: float = 0.5):
    """For each (regime,time) plot truth h(x) + excluded band and compare FV shock vs PINN-traj shock."""
    nR = len(regimes)
    nT = len(times)
    # Two rows per regime: h on row 2*i, q on row 2*i+1 (q = h*u)
    fig, axes = plt.subplots(2*nR, nT, figsize=(4.2*nT, 2.8*2*nR), squeeze=False)

    for i, reg in enumerate(regimes):
        hL, hR, shape = float(reg[0]), float(reg[1]), str(reg[2])
        for j, t in enumerate(times):
            ax_h = axes[2*i][j]
            ax_q = axes[2*i + 1][j]
            # FV truth for overlay only; keep Nx/dt coarse for speed (does not need high fidelity).
            dt = float(dt)
            snaps = simulate_shallow_water(
                h_left=hL, h_right=hR, dam_shape=shape,
                Lx=Lx, Nx=Nx, t_final=float(t), dt=dt,
                sample_interval=float(t), plot=False
            )
            snap = snaps[-1]
            x = np.asarray(snap["x"], dtype=float)
            h = np.asarray(snap["h"], dtype=float)
            u = np.asarray(snap["u"], dtype=float)
            q = h * u

            y_loc = h if str(indicator).lower() == "h" else q
            x_fv = _estimate_shock_x_from_truth_snapshot(x, y_loc, indicator=indicator)

            # ---- h panel ----
            ax_h.plot(x, h, "k-", lw=1.6, label="truth h")
            ax_h.axvline(x_fv, color="r", ls="--", lw=1.3, label="FV shock (argmax|dh/dx|)")

            if float(hR) < float(dry_hR_eps):
                ax_h.text(0.02, 0.88, "dry-bed: no exclusion", transform=ax_h.transAxes,
                          ha="left", va="top", fontsize=9, color="orange")
            else:
                ax_h.axvspan(max(0.0, x_fv - w_excl), min(Lx, x_fv + w_excl),
                             color="orange", alpha=0.22, label="excluded band")

            ax_h.set_title(f"h: hL={hL:g}, hR={hR:g}, t={float(t):g} (center=FV)")
            ax_h.set_xlabel("x")
            ax_h.set_ylabel("h")
            ax_h.grid(True, alpha=0.25)

            # ---- q panel ----
            ax_q.plot(x, q, "k-", lw=1.6, label="truth q")
            ax_q.axvline(x_fv, color="r", ls="--", lw=1.3, label="FV shock")

            if float(hR) < float(dry_hR_eps):
                ax_q.text(0.02, 0.88, "dry-bed: no exclusion", transform=ax_q.transAxes,
                          ha="left", va="top", fontsize=9, color="orange")
            else:
                ax_q.axvspan(max(0.0, x_fv - w_excl), min(Lx, x_fv + w_excl),
                             color="orange", alpha=0.22, label="excluded band")

            ax_q.set_title(f"q: hL={hL:g}, hR={hR:g}, t={float(t):g} (center=FV)")
            ax_q.set_xlabel("x")
            ax_q.set_ylabel("q")
            ax_q.grid(True, alpha=0.25)

    # Collect legend entries from all axes (some panels may not include PINN shock_traj).
    all_handles = []
    all_labels = []
    for row in axes:
        for ax in row:
            hh, ll = ax.get_legend_handles_labels()
            all_handles.extend(hh)
            all_labels.extend(ll)

    # De-duplicate while preserving order
    uniq = {}
    for h, l in zip(all_handles, all_labels):
        if l not in uniq:
            uniq[l] = h

    # Place legend slightly below the very top and put the title above it.
    fig.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper center",
        ncol=min(4, len(uniq)),
        frameon=True,
        bbox_to_anchor=(0.5, 0.975),
        fontsize=10,
    )
    fig.suptitle("Shock-aware sanity: excluded band vs truth shock", y=0.998, fontsize=12)

    # Reserve enough space at the top so the legend doesn't overlap with subplots.
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    # Save under a dedicated sanity subfolder to keep the experiment folder clean.
    sanity_dir = os.path.join(pinn_obj.checkpoint_folder, "sanity")
    os.makedirs(sanity_dir, exist_ok=True)
    out_path = os.path.join(sanity_dir, save_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _rms(z: torch.Tensor) -> float:
    """Return the root-mean-square magnitude of a tensor as a Python float."""
    return float(torch.sqrt(torch.mean(z.detach() ** 2) + 1e-12).cpu().item())


def _run_single_pipeline(run_args: argparse.Namespace, root_cwd: str) -> str:
    """Run one full experiment (data->train->eval). Returns the experiment checkpoint folder."""
    # Preserve and restore CWD because this script uses chdir.
    cwd0 = os.getcwd()
    try:
        # make directories
        os.makedirs(run_args.output_dir, exist_ok=True)
        os.chdir(run_args.output_dir)

        # reproducibility
        set_global_seed(run_args.seed)
        pipeline_timing = TimingRecorder()
        _pipeline_start = time.perf_counter()

        # capture command-line for later
        cmd_local = " ".join(sys.argv)

        # build the model
        pinn_local = Linear_PINN(
            architecture = {"model":run_args.model, "layers":run_args.layers},
            activation = getattr(nn, run_args.activation)(),
            num_epochs = run_args.num_epochs,
            optimizer = optim.Adam,
            optimizer_lr = run_args.optimizer_lr,
            data_weight = run_args.data_weight,
            pde_weight = run_args.pde_weight,
            l2 = {'data': run_args.l2_data, 'pde': run_args.l2_pde},
            bc_weight = run_args.bc_weight,
            ic_weight = run_args.ic_weight,
            nonneg_h_weight = run_args.nonneg_h_weight,
            lambda_q = run_args.lambda_q,
            enforce_pos_h = run_args.enforce_pos_h,
            rampup_epochs_share = run_args.rampup_epochs_share,
            scheduler_params = {"status":not run_args.disable_scheduler, "T_max":run_args.scheduler_T_max, "eta_min":run_args.scheduler_eta_min},
            grad_clip = run_args.grad_clip,
            display = run_args.display,
            device = run_args.device,
        )

        # write out a human-readable record in <output_dir>/experiments/<timestamp>/experiment_settings.txt
        with open(os.path.join(pinn_local.checkpoint_folder, "experiment_settings.txt"), "a") as f:
            f.write("\n\n=== COMMAND LINE ===\n")
            f.write(cmd_local)
            f.write("\n\n=== PARSED ARGS ===\n")
            json.dump(vars(run_args), f, indent=2)

        # Reproducibility artifacts: resolved config, command, environment and source snapshot.
        try:
            save_run_manifest(pinn_local.checkpoint_folder, run_args, command=sys.argv)
            snapshot_sources(SCRIPT_DIR, os.path.join(pinn_local.checkpoint_folder, "scripts"))
        except Exception as e:
            print("Could not save reproducibility artifacts:", e)

        if run_args.load_checkpoint:
            with pipeline_timing.measure("checkpoint_load"):
                pinn_local.load_checkpoint(run_args.load_checkpoint)
        else:
            # generate data
            _data_start = time.perf_counter()
            pinn_local.get_data(
                train_regimes = run_args.train_regime,
                train_sample_interval = run_args.train_sample_interval,
                val_regime = run_args.val_regime,
                val_sample_interval = run_args.val_sample_interval,
                Lx = run_args.Lx,
                Nx = run_args.Nx,
                t_final = run_args.t_final,
                dt = run_args.dt,
                tol = run_args.tol,
                N_colloc_train = run_args.N_colloc_train,
                N_colloc_val = run_args.N_colloc_val,
                shock_aware_colloc = {
                    "status": bool(run_args.shock_aware_colloc),
                    "indicator": run_args.shock_indicator,
                    "w_excl": float(run_args.shock_w_excl),
                    "w_focus": float(run_args.shock_w_focus),
                    "focus_share": float(run_args.shock_focus_share),
                    "dry_hR_eps": float(run_args.dry_hR_eps),
                    "pde_gate": bool(run_args.pde_gate),
                    "gate_alpha_grad": float(run_args.gate_alpha_grad),
                    "gate_h_min": float(run_args.gate_h_min),
                    "gate_floor": float(run_args.gate_floor),
                    "shock_smooth_window": int(run_args.shock_smooth_window),
                    "shock_boundary_buf_frac": float(run_args.shock_boundary_buf_frac),
                    "residual_scaling": str(run_args.residual_scaling),
                    "residual_ema_beta": float(run_args.residual_ema_beta),
                    "residual_scale_eps": float(run_args.residual_scale_eps),
                    # Dry-bed loss balancing (loss-only)
                    "balance_dry_loss": bool(getattr(run_args, "balance_dry_loss", False)),
                    "dry_loss_weight": float(getattr(run_args, "dry_loss_weight", 3.0)),
                }
            )
            pipeline_timing.add("data_generation_and_preparation", time.perf_counter() - _data_start)

            # Optional: shock-aware sanity overlay (truth shock + excluded band).
            if getattr(run_args, "shock_sanity_overlay", False):
                regs = list(run_args.shock_sanity_overlay_regimes) if run_args.shock_sanity_overlay_regimes is not None else []
                if len(regs) == 0:
                    # default: first train regime, last train regime, and val regime
                    try:
                        if isinstance(run_args.train_regime, (list, tuple)) and len(run_args.train_regime) > 0:
                            regs.append(run_args.train_regime[0])
                            regs.append(run_args.train_regime[-1])
                    except Exception:
                        pass
                    regs.append(run_args.val_regime)
                # de-dup
                regs = [(float(r[0]), float(r[1]), str(r[2])) for r in regs]
                regs = list(dict.fromkeys(regs))

                times = [float(t) for t in run_args.shock_sanity_overlay_times] if run_args.shock_sanity_overlay_times is not None else []
                if len(times) == 0:
                    tmax = float(run_args.t_final)
                    times = [min(0.5 * tmax, tmax), tmax]
                times = [t for t in times if 0.0 < float(t) <= float(run_args.t_final)]
                times = sorted(list({float(t) for t in times}))

                try:
                    outp = _shock_exclusion_overlay(
                        pinn_obj=pinn_local,
                        regimes=regs,
                        times=times,
                        save_name="shock_exclusion_overlay.png",
                        Lx=float(run_args.Lx),
                        Nx=int(run_args.shock_sanity_overlay_nx),
                        dt=float(run_args.shock_sanity_overlay_dt),
                        t_final=float(run_args.t_final),
                        w_excl=float(run_args.shock_w_excl),
                        indicator=str(run_args.shock_indicator),
                        dry_hR_eps=float(run_args.dry_hR_eps),
                    )
                    print(f"[sanity] Saved shock exclusion overlay to {outp}")
                except Exception as e:
                    print("[sanity] shock overlay failed:", e)

            # --- BEGIN: Inserted missing sanity diagnostics ---
            # Quick shock-aware sampling sanity plot (fast). Saved under <checkpoint>/sanity/.
            if bool(run_args.shock_aware_colloc):
                try:
                    sanity_regimes = []
                    if isinstance(run_args.train_regime, (list, tuple)) and len(run_args.train_regime) > 0:
                        sanity_regimes.append(run_args.train_regime[0])
                        if len(run_args.train_regime) > 1:
                            sanity_regimes.append(run_args.train_regime[-1])
                    else:
                        sanity_regimes = [run_args.val_regime]

                    tmax = float(run_args.t_final)
                    sanity_times = [min(0.2 * tmax, tmax), min(0.5 * tmax, tmax), tmax]
                    sanity_times = sorted(list({float(t) for t in sanity_times if float(t) > 0.0}))

                    pinn_local.shock_sampling_sanity_plot(
                        regimes=sanity_regimes,
                        times=sanity_times,
                        n_per_time=2000,
                        save_name="shock_sampling_sanity.png",
                    )
                except Exception as e:
                    print("[sanity] shock_sampling_sanity_plot failed:", e)

            # ---- PDE gating sanity (only in sanity mode) ----
            if bool(run_args.sanity) and bool(run_args.shock_aware_colloc) and hasattr(pinn_local, "_pde_gating_weights"):
                try:
                    cfg = pinn_local.get_data_params.get("shock_aware_colloc", {}) if hasattr(pinn_local, "get_data_params") else {}
                    x = pinn_local.colloc["train"]["x"]
                    t = pinn_local.colloc["train"]["t"]
                    pparam = pinn_local.colloc["train"]["param"]

                    # compute with gradients enabled for h_x inside gating
                    w = pinn_local._pde_gating_weights(x, t, pparam, cfg=cfg).detach().cpu().numpy().reshape(-1)

                    sanity_dir = os.path.join(pinn_local.checkpoint_folder, "sanity")
                    os.makedirs(sanity_dir, exist_ok=True)

                    stats_path = os.path.join(sanity_dir, "pde_gating_stats.txt")
                    with open(stats_path, "w") as f:
                        f.write("PDE gating sanity (weights in [gate_floor,1])\n")
                        f.write(json.dumps(cfg, indent=2))
                        f.write("\n\n")
                        f.write(f"min={float(np.min(w)):.6g}\n")
                        f.write(f"max={float(np.max(w)):.6g}\n")
                        f.write(f"mean={float(np.mean(w)):.6g}\n")
                        f.write(f"p10={float(np.quantile(w, 0.10)):.6g}\n")
                        f.write(f"p50={float(np.quantile(w, 0.50)):.6g}\n")
                        f.write(f"p90={float(np.quantile(w, 0.90)):.6g}\n")

                    fig = plt.figure(figsize=(6.0, 3.6))
                    plt.hist(w, bins=40)
                    plt.title("PDE gating weights histogram")
                    plt.xlabel("w_gate")
                    plt.ylabel("count")
                    plt.grid(True, alpha=0.25)
                    try:
                        fig.tight_layout()
                    except Exception:
                        pass
                    out_png = os.path.join(sanity_dir, "pde_gating_hist.png")
                    try:
                        fig.savefig(out_png, dpi=200)
                    except RecursionError as e:
                        print(f"[warn] savefig recursion error (pde_gating_hist) -> skipping: {e}")
                    plt.close(fig)
                    print(f"[sanity] wrote {stats_path} and pde_gating_hist.png")
                except Exception as e:
                    print("[sanity] PDE gating sanity failed:", e)

            # ---- Dry-bed loss balancing sanity (only in sanity mode) ----
            if bool(run_args.sanity) and bool(getattr(run_args, "balance_dry_loss", False)):
                try:
                    sanity_dir = os.path.join(pinn_local.checkpoint_folder, "sanity")
                    os.makedirs(sanity_dir, exist_ok=True)

                    # Prefer masks created by pinn.get_data
                    n_train = None
                    n_dry = None
                    frac = None
                    if hasattr(pinn_local, "data") and isinstance(getattr(pinn_local, "data"), dict):
                        dtr = pinn_local.data.get("train", {})
                        if isinstance(dtr, dict) and ("is_dry" in dtr):
                            m = dtr["is_dry"].detach().cpu().numpy().reshape(-1)
                            n_train = int(m.size)
                            n_dry = int((m > 0.5).sum())
                            frac = float(n_dry) / float(max(1, n_train))

                    w_dry = float(getattr(run_args, "dry_loss_weight", 3.0))
                    outp = os.path.join(sanity_dir, "dry_bed_loss_balance_stats.txt")
                    with open(outp, "w") as f:
                        f.write("Dry-bed loss balancing sanity\n")
                        f.write(f"balance_dry_loss={bool(getattr(run_args, 'balance_dry_loss', False))}\n")
                        f.write(f"dry_loss_weight={w_dry}\n")
                        if n_train is not None:
                            f.write(f"n_train={n_train}\n")
                            f.write(f"n_dry={n_dry}\n")
                            f.write(f"dry_fraction={frac:.6g}\n")
                            # effective mean weight under this dataset composition
                            eff_mean_w = (1.0 - frac) * 1.0 + frac * w_dry
                            f.write(f"effective_mean_weight={eff_mean_w:.6g}\n")
                        else:
                            f.write("is_dry mask not found in pinn_local.data['train'] (check pinn.get_data)\n")

                    print(f"[sanity] wrote {outp}")
                except Exception as e:
                    print("[sanity] dry-bed loss balancing sanity failed:", e)

            # ---- PDE residual scaling sanity (only in sanity mode) ----
            if bool(run_args.sanity) and hasattr(pinn_local, "pde_residual"):
                try:
                    cfg = pinn_local.get_data_params.get("shock_aware_colloc", {}) if hasattr(pinn_local, "get_data_params") else {}
                    mode = str(cfg.get("residual_scaling", "none"))

                    x_all = pinn_local.colloc["train"]["x"]
                    t_all = pinn_local.colloc["train"]["t"]
                    p_all = pinn_local.colloc["train"]["param"]
                    n = int(min(5000, x_all.shape[0]))
                    idx = torch.randperm(x_all.shape[0], device=x_all.device)[:n]
                    x = x_all[idx]
                    t = t_all[idx]
                    pparam = p_all[idx]

                    r1, r2 = pinn_local.pde_residual(x, t, pparam)

                    if hasattr(pinn_local, "_normalize_pde_residuals"):
                        r1n, r2n, scales = pinn_local._normalize_pde_residuals(r1, r2, cfg=cfg)
                    else:
                        r1n, r2n, scales = r1, r2, {"s1": 1.0, "s2": 1.0}

                    def _rms(z: torch.Tensor) -> float:
                        return float(torch.sqrt(torch.mean(z.detach() ** 2) + 1e-12).cpu().item())

                    stats = {
                        "residual_scaling": mode,
                        "n_points": int(n),
                        "r1_rms": _rms(r1),
                        "r2_rms": _rms(r2),
                        "r1n_rms": _rms(r1n),
                        "r2n_rms": _rms(r2n),
                        "scale_s1": float(scales.get("s1", 1.0)),
                        "scale_s2": float(scales.get("s2", 1.0)),
                    }

                    sanity_dir = os.path.join(pinn_local.checkpoint_folder, "sanity")
                    os.makedirs(sanity_dir, exist_ok=True)
                    stats_path = os.path.join(sanity_dir, "pde_residual_scaling_stats.txt")
                    with open(stats_path, "w") as f:
                        f.write("PDE residual scaling sanity\n")
                        f.write(json.dumps(cfg, indent=2))
                        f.write("\n\n")
                        f.write(json.dumps(stats, indent=2))
                        f.write("\n")

                    r1a = torch.log10(torch.abs(r1.detach()).clamp_min(1e-12)).cpu().numpy().reshape(-1)
                    r2a = torch.log10(torch.abs(r2.detach()).clamp_min(1e-12)).cpu().numpy().reshape(-1)
                    r1na = torch.log10(torch.abs(r1n.detach()).clamp_min(1e-12)).cpu().numpy().reshape(-1)
                    r2na = torch.log10(torch.abs(r2n.detach()).clamp_min(1e-12)).cpu().numpy().reshape(-1)

                    fig = plt.figure(figsize=(7.0, 4.2))
                    plt.hist(r1a, bins=40, alpha=0.6, label="log10|r1|")
                    plt.hist(r2a, bins=40, alpha=0.6, label="log10|r2|")
                    plt.legend()
                    plt.title("Raw PDE residual magnitudes")
                    plt.xlabel("log10(|residual|)")
                    plt.ylabel("count")
                    plt.grid(True, alpha=0.25)
                    try:
                        fig.tight_layout()
                    except Exception:
                        pass
                    out_png = os.path.join(sanity_dir, "pde_residual_raw_hist.png")
                    try:
                        fig.savefig(out_png, dpi=200)
                    except RecursionError as e:
                        print(f"[warn] savefig recursion error (pde_residual_raw_hist) -> skipping: {e}")
                    plt.close(fig)

                    fig = plt.figure(figsize=(7.0, 4.2))
                    plt.hist(r1na, bins=40, alpha=0.6, label="log10|r1| (scaled)")
                    plt.hist(r2na, bins=40, alpha=0.6, label="log10|r2| (scaled)")
                    plt.legend()
                    plt.title("Scaled PDE residual magnitudes")
                    plt.xlabel("log10(|residual|)")
                    plt.ylabel("count")
                    plt.grid(True, alpha=0.25)
                    try:
                        fig.tight_layout()
                    except Exception:
                        pass
                    out_png = os.path.join(sanity_dir, "pde_residual_scaled_hist.png")
                    try:
                        fig.savefig(out_png, dpi=200)
                    except RecursionError as e:
                        print(f"[warn] savefig recursion error (pde_residual_scaled_hist) -> skipping: {e}")
                    plt.close(fig)

                    print(f"[sanity] wrote {stats_path} and pde_residual_*_hist.png")
                except Exception as e:
                    print("[sanity] PDE residual scaling sanity failed:", e)
            # --- END: Inserted missing sanity diagnostics ---

            # In sanity mode we want a fast smoke-test.
            if run_args.sanity:
                print("[sanity] Running a tiny training to exercise gating/scaling...")
                pinn_local.train()
                pinn_local.evaluate(
                    test_t=run_args.test_t[-1],
                    test_h=run_args.test_h[0],
                    plot=True,
                )

                pinn_local.plot_trom_style_final_profiles(
                    test_h=[
                        (12.0, 0.0, "perpendicular"),
                        (12.0, 7.0, "perpendicular"),
                        (15.0, 4.0, "perpendicular"),
                        (18.0, 0.0, "perpendicular"),
                        (26.0, 0.14, "perpendicular"),
                        (26.0, 7.0, "perpendicular"),
                    ],
                    test_t=run_args.test_t[-1],
                    field="h",
                    plot_kind="spectrum",
                    save_name="sanity_trom_spectrum_h.png",
                )
                print("[sanity] Done (trained briefly + wrote sanity artifacts). Exiting.")
                return pinn_local.checkpoint_folder

            # train & then test
            pinn_local.train()

        # Generic evaluation plan (preferred for new experiments). Legacy CLI evaluation
        # remains supported below for exact backward compatibility.
        if run_args.eval_config_json:
            eval_path = _resolve_path(run_args.eval_config_json, root_cwd)
            with open(eval_path, "r") as f:
                eval_plan = json.load(f)
            _eval_start = time.perf_counter()
            artifacts = run_evaluation_plan(pinn_local, eval_plan)
            pipeline_timing.add("evaluation_total", time.perf_counter() - _eval_start)
            dump_json(os.path.join(pinn_local.checkpoint_folder, "evaluation_artifacts.json"), artifacts)
            pipeline_timing.add("pipeline_total", time.perf_counter() - _pipeline_start)
            pipeline_timing.save(os.path.join(pinn_local.checkpoint_folder, "pipeline_timing.json"))
            return pinn_local.checkpoint_folder

        suffix_local = "paper" if (run_args.paper_eval_grid_hL is None and run_args.paper_eval_grid_hR is None) else "specific"
        pinn_local.test_model(
            test_t=run_args.test_t,
            test_h=run_args.test_h,
            plot=run_args.display,
            paper_eval_kwargs={
                "sampling":run_args.paper_eval_sampling,
                "n_samples":run_args.paper_eval_n_sample,
                "seed":run_args.paper_eval_seed,
                "grid_hL":run_args.paper_eval_grid_hL,
                "grid_hR":run_args.paper_eval_grid_hR,
                "dam_shape":run_args.paper_eval_dam_shape,
                "include_H1":not run_args.paper_eval_exclude_H1,
                "Nx":run_args.paper_eval_Nx,
                "t_final":run_args.paper_eval_t_final,
                "dt":run_args.paper_eval_dt,
                "sample_interval":run_args.paper_eval_sample_interval,
                "fields": [s.strip() for s in run_args.paper_eval_fields.split(",") if s.strip()],
                "aggregate": run_args.paper_eval_aggregate,
                "save_name":run_args.paper_eval_save_name if run_args.paper_eval_save_name is not None else f"eval_{run_args.paper_eval_sampling}_{suffix_local}"
            } if run_args.paper_eval else None
        )

        if run_args.plot_special_t2p5:
            special_regime = [
                (26.0, 0.14, "perpendicular"),
            ]

            for field in ("h", "q"):
                for plot_kind in ("profile", "spectrum"):
                    pinn_local.plot_trom_style_final_profiles(
                        test_h=special_regime,
                        test_t=2.5,
                        field=field,
                        plot_kind=plot_kind,
                    )

        if run_args.slice_eval:
            hL0 = float(run_args.slice_eval_hL)
            hR_list = np.linspace(float(run_args.slice_eval_hR_min), float(run_args.slice_eval_hR_max), int(run_args.slice_eval_nhR))
            slice_fields = [s.strip() for s in run_args.slice_eval_fields.split(",") if s.strip()]

            pinn_local.paper_style_eval(
                sampling="grid",
                grid_hL=[hL0],
                grid_hR=[float(x) for x in hR_list],
                dam_shape=run_args.paper_eval_dam_shape,
                include_H1=not run_args.paper_eval_exclude_H1,
                Nx=run_args.paper_eval_Nx,
                t_final=run_args.paper_eval_t_final,
                dt=run_args.paper_eval_dt,
                sample_interval=run_args.paper_eval_sample_interval,
                fields=slice_fields,
                aggregate=run_args.paper_eval_aggregate,
                save_name=f"slice_hL={hL0:g}_hR_{run_args.slice_eval_nhR}_{suffix_local}"
            )

        pipeline_timing.add("pipeline_total", time.perf_counter() - _pipeline_start)
        pipeline_timing.save(os.path.join(pinn_local.checkpoint_folder, "pipeline_timing.json"))
        return pinn_local.checkpoint_folder
    finally:
        os.chdir(cwd0)


def apply_config_json(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    root_cwd: str,
) -> argparse.Namespace:
    """Load a general experiment config, while letting explicit CLI options override it."""
    if args.config is None:
        return args

    config_path = _resolve_path(args.config, root_cwd)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    valid_keys = {action.dest for action in parser._actions}
    unknown = set(config) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown configuration option(s) in {config_path}: "
            + ", ".join(sorted(unknown))
        )

    # Explicit CLI arguments take precedence over values in the JSON config.
    explicit = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            explicit.add(token[2:].split("=", 1)[0].replace("-", "_"))

    for key, value in config.items():
        if key not in explicit:
            setattr(args, key, value)

    return args


def main():
    '''
    Example to run:

    nohup python run_experiments.py --N_colloc_train 63000 --N_colloc_val 11000 &

    # Shock-aware collocation sampling (recommended for dam-break):
    # nohup python run_experiments.py --shock_aware_colloc --shock_w_excl 1.5 --shock_w_focus 6.0 --shock_focus_share 0.6 --N_colloc_train 63000 --N_colloc_val 11000 &
    '''
    p = argparse.ArgumentParser()
    # general
    p.add_argument("--output_dir", type=str, default="./", help="where to write logs/experiments/plots")
    p.add_argument("--display", action="store_true", help="enable all IPython.display() and plt.show() calls")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["legacy_auto", "cpu", "cuda", "mps"], default="legacy_auto",
                   help="Compute device. legacy_auto preserves submitted behavior: CUDA if available, otherwise CPU.")
    p.add_argument("--sanity", action="store_true", help="run a quick smoke test with reduced spatial/data settings")
    p.add_argument("--sanity_epochs", type=int, default=1, help="epochs used by --sanity (default: 1)")
    p.add_argument("--config", type=str, default=None,
        help="JSON file containing experiment/training arguments. Explicit command-line arguments override values from this file.",
    )

    # greedy search / multi-run launcher
    p.add_argument("--greedy_search", action="store_true",
                   help="Run multiple experiments defined in --greedy_json. Each experiment runs the full train/eval flow in its own folder and a summary CSV is produced.")
    p.add_argument("--greedy_json", type=str, default=None,
                   help="Path to JSON describing experiments. Format: { 'baseline': {...}, 'experiments': [ {'name': 'exp1', 'overrides': {...}}, ... ] }.")
    p.add_argument("--greedy_name", type=str, default=None,
                   help="Name of the greedy sweep folder under output_dir. Default: greedy_<timestamp>.")
    p.add_argument("--greedy_max_runs", type=int, default=None,
                   help="Optional cap on number of experiments to run (for quick tests).")
    p.add_argument("--greedy_run_baseline", action="store_true", default=True,
                   help="In greedy_search mode, run an additional baseline experiment (baseline overrides only) named 'baseline' before other experiments.")
    p.add_argument("--no_greedy_run_baseline", dest="greedy_run_baseline", action="store_false",
                   help="Disable the automatic baseline run in greedy_search mode.")

    # model architecture & training
    p.add_argument("--model", choices=["lin", "residual_lin"], default="lin")
    p.add_argument("--layers", type=int, nargs="+", default=[4, 120, 120, 120, 120, 120, 2], 
                   help="nargs=+ lets you pass things like --layers 4 120 120 2")
    p.add_argument("--activation", choices=["SiLU", "ReLU", "Tanh", "Identity"], default="SiLU")
    p.add_argument("--num_epochs", type=int, default=12000)
    p.add_argument("--optimizer_lr", type=float, default=1e-3)
    p.add_argument("--data_weight", type=float, default=1.0)
    p.add_argument("--l2_data", action="store_true")
    p.add_argument("--pde_weight", type=float, default=0.1)
    p.add_argument("--l2_pde", action="store_true")
    p.add_argument("--bc_weight", type=float, default=1.0)
    p.add_argument("--ic_weight", type=float, default=1.0, help="Weight for analytic initial-condition loss at t=0.")
    p.add_argument('--nonneg_h_weight', type=float, default=0.0, help='Weight for non-negativity penalty on predicted depth h.')
    p.add_argument("--lambda_q", type=float, default=1.0, help="Relative weight for q vs h in standardized data + BC losses (lambda_q).")
    p.add_argument("--enforce_pos_h", type=str, default="none", choices=["none", "softplus"], 
                   help="Depth parameterization. 'softplus' enforces h>0 by applying softplus to the network h-output.")
    p.add_argument("--rampup_epochs_share", type=float, default=0.0)
    p.add_argument("--disable_scheduler", action="store_true", help="disable the LR scheduler (CosineAnnealingLR)")
    p.add_argument("--scheduler_T_max", type=int, default=12000)
    p.add_argument("--scheduler_eta_min", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # data generation
    p.add_argument("--train_regime", type=parse_regime, 
                   nargs="+", default=[(10.0, 0.0, "perpendicular"), (13.0, 0.0, "perpendicular"), (16.0, 0.0, "perpendicular"), 
                                       (19.0, 0.0, "perpendicular"), (22.0, 0.0, "perpendicular"), (25.0, 0.0, "perpendicular"),
                                       (28.0, 0.0, "perpendicular"), (10.0, 4.0, "perpendicular"), (13.0, 4.0, "perpendicular"),
                                       (16.0, 4.0, "perpendicular"), (19.0, 4.0, "perpendicular"), (22.0, 4.0, "perpendicular"),
                                       (25.0, 4.0, "perpendicular"), (28.0, 4.0, "perpendicular"), (10.0, 8.0, "perpendicular"), 
                                       (13.0, 8.0, "perpendicular"), (16.0, 8.0, "perpendicular"), (19.0, 8.0, "perpendicular"),
                                       (22.0, 8.0, "perpendicular"), (25.0, 8.0, "perpendicular"), (28.0, 8.0, "perpendicular")],
                   help="use multiple like: --train_regime 10,5,perpendicular 15,5,perpendicular 20,5,perpendicular")
    p.add_argument("--train_sample_interval", type=float, default=0.25)
    p.add_argument("--val_regime", type=parse_regime, default=(17.0, 4.0, "perpendicular"))
    p.add_argument("--val_sample_interval", type=float, default=0.25)
    p.add_argument("--Lx", type=int, default=100)
    p.add_argument("--Nx", type=int, default=402)
    p.add_argument("--t_final", type=float, default=2.5)
    p.add_argument("--dt", type=float, default=0.0001)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--N_colloc_train", type=int, default=None)
    p.add_argument("--N_colloc_val", type=int, default=None)
    p.add_argument("--train_grid_json", type=str, help="Path to JSON with keys: h_L (list), h_R (list), shape (str) and etc for training")

    # shock-aware collocation sampling (recommended for dam-break)
    p.add_argument("--shock_aware_colloc", action="store_true", help="Use regime/time-dependent shock-aware collocation sampling (excludes a band around shock and oversamples nearby).")
    p.add_argument("--shock_indicator", choices=["h","q"], default="h", help="Field used to locate shock: h or q (q=h*u from truth data).")
    p.add_argument("--shock_w_excl", type=float, default=1.5,
                   help="Exclude PDE collocation points within |x-x_shock(t)| < w_excl (hard exclusion band).")
    p.add_argument("--shock_w_focus", type=float, default=6.0,
                   help="Near-shock focus band half-width (sample extra PDE points in [w_excl, w_excl+w_focus] around x_shock).")
    p.add_argument("--shock_focus_share", type=float, default=0.6,
                   help="Fraction of PDE collocations drawn from the near-shock focus band (rest from uniform smooth region).")

    # PDE gating (softly downweight PDE residual in steep / near-dry regions)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pde_gate", dest="pde_gate", action="store_true", help="Enable PDE gating weights (downweight PDE loss near steep gradients / near-dry).")
    g.add_argument("--no_pde_gate", dest="pde_gate", action="store_false", help="Disable PDE gating weights.")
    p.set_defaults(pde_gate=True)
    p.add_argument("--gate_alpha_grad", type=float, default=8.0,
                   help="Gating sharpness vs |h_x|: larger => stronger downweighting of PDE loss in steep regions.")
    p.add_argument("--gate_h_min", type=float, default=0.05,
                   help="Downweight PDE when predicted h is near dry-bed (h < gate_h_min).")
    p.add_argument("--gate_floor", type=float, default=0.05,
                   help="Minimum PDE gating weight (lower bound on PDE loss multiplier).")
    p.add_argument("--dry_hR_eps", type=float, default=0.5,
                   help="If hR < dry_hR_eps treat regime as dry-bed and skip shock exclusion/focus.")
    p.add_argument("--shock_smooth_window", type=int, default=7,
                   help="Odd moving-average window used to smooth truth h(x) before locating shock (reduces spurious peaks).")
    p.add_argument("--shock_boundary_buf_frac", type=float, default=0.02,
                   help="Ignore this fraction of cells near x=0 and x=L when locating shock (avoids boundary artifacts).")

    # PDE residual scaling (normalize PDE residual components so one equation doesn't dominate)
    p.add_argument(
        "--residual_scaling",
        type=str,
        default="none",
        choices=["none", "batch_rms", "ema_rms"],
        help=("Normalize PDE residual components before applying gating/aggregation. "
              "none: no scaling; batch_rms: divide by batch RMS (noisy); "
              "ema_rms: divide by EMA of RMS (stable)."),
    )
    p.add_argument(
        "--residual_ema_beta",
        type=float,
        default=0.99,
        help="EMA beta used when residual_scaling=ema_rms (closer to 1 => smoother scale).",
    )
    p.add_argument(
        "--residual_scale_eps",
        type=float,
        default=1e-8,
        help="Epsilon for residual RMS to avoid division by zero.",
    )


    # Dry-bed loss balancing (loss-only; dry-bed defined strictly as hR==0 inside pinn)
    p.add_argument("--balance_dry_loss", action="store_true",
                   help="Reweight supervised data+BC losses for dry-bed regimes (strict hR==0) without changing sampling.")
    p.add_argument("--dry_loss_weight", type=float, default=3.0,
                   help="Multiplicative loss weight for dry-bed samples when --balance_dry_loss is enabled.")


    # shock-aware sanity: overlay excluded band vs. truth shock location
    p.add_argument("--shock_sanity_overlay", action="store_true",
                   help="Overlay the excluded shock band on top of the FV truth h(x), using FV shock (argmax|dh/dx|) as the band center (no PINN shock_traj).")
    p.add_argument("--shock_sanity_overlay_regimes", type=parse_regime, nargs="+", default=None,
                   help="Regimes for overlay sanity, e.g. --shock_sanity_overlay_regimes 28,0,perpendicular 10,8,perpendicular. If omitted, uses [first train_regime, last train_regime, val_regime].")
    p.add_argument("--shock_sanity_overlay_times", type=float, nargs="+", default=None,
                   help="Times for overlay sanity (must be within t_final). If omitted, uses [0.5*t_final, t_final].")
    p.add_argument("--shock_sanity_overlay_nx", type=int, default=201,
                   help="Nx for FV truth used in overlay sanity. Default 201 (fast).")
    p.add_argument("--shock_sanity_overlay_dt", type=float, default=5e-4,
                   help="dt for FV truth used in overlay sanity. Default 5e-4 (fast).")

    # test / eval
    p.add_argument("--eval_paper_figures", action="store_true", help="Evaluate/plot the same parameter pairs used in ROM paper figures (Figures 5-8).")
    p.add_argument("--load_checkpoint", type=str, default=None, help="Path to .pth. If set, skip get_data() and training, just evaluate.")
    p.add_argument("--eval_config_json", type=str, default=None,
                   help="Optional generic JSON evaluation plan. When provided, it replaces legacy test/paper/slice evaluation branching.")
    p.add_argument(
        "--plot_special_t2p5",
        action="store_true",
        help="Additionally plot hL=26, hR=0.14 at t=2.5.",
    )
    p.add_argument("--test_t", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    p.add_argument("--test_h", type=parse_regime, nargs="+", default=[(8.0,0.0,"perpendicular"), (14.0,0.0,"perpendicular"), (20.5,0.0,"perpendicular"), 
                                                                      (30.0,0.0,"perpendicular"), (8.0,2.0,"perpendicular"), (14.0,2.0,"perpendicular"), 
                                                                      (20.5,2.0,"perpendicular"), (30.0,2.0,"perpendicular"), (9.5,8.5,"perpendicular"), 
                                                                      (14.0,8.5,"perpendicular"), (20.5,8.5,"perpendicular"), (30, 8.5,"perpendicular"),
                                                                      (25.0,0.05,"perpendicular"), (25.0,0.3,"perpendicular"), (25.0,2.0,"perpendicular")])
    p.add_argument("--paper_eval", action="store_true", help="Run space-time relative error eval like the paper")
    # grid/time parameters used for the paper-style eval
    p.add_argument("--paper_eval_sampling", choices=["grid", "mc"], default="mc")
    p.add_argument("--paper_eval_n_sample", type=int, default=99, help="used if sampling=mc")
    p.add_argument("--paper_eval_seed", type=int, default=123)
    p.add_argument("--paper_eval_exclude_H1", action="store_true")
    p.add_argument("--paper_eval_grid_hL", type=float, nargs="+", default=None)
    p.add_argument("--paper_eval_grid_hR", type=float, nargs="+", default=None)
    p.add_argument("--paper_eval_dam_shape", type=str, default='perpendicular')
    p.add_argument("--paper_eval_Nx", type=int, default=402)
    p.add_argument("--paper_eval_t_final", type=float, default=2.5)
    p.add_argument("--paper_eval_dt", type=float, default=1e-4)
    p.add_argument("--paper_eval_sample_interval", type=float, default=0.01)
    p.add_argument("--paper_eval_grid_json", type=str, help="Path to JSON with keys: h_L (list), h_R (list) and etc for evaluation")
    p.add_argument("--paper_eval_save_name", type=str)

    # paper-style eval extensions
    p.add_argument("--paper_eval_fields", type=str, default="h,q,state", help="Comma-separated fields to evaluate in paper_eval: h,q. Example: --paper_eval_fields h,q")
    p.add_argument("--paper_eval_aggregate", choices=["mean","integral"], default="mean", help="Aggregate across parameter samples for 'avg': mean (paper-style) or integral (area*mean).")

    # fixed-hL slice eval (Table-4/5 style)
    p.add_argument("--slice_eval", action="store_true", help="Run a fixed-hL sweep over hR and compute paper-style aggregates.")
    p.add_argument("--slice_eval_hL", type=float, default=25.0)
    p.add_argument("--slice_eval_hR_min", type=float, default=0.0)
    p.add_argument("--slice_eval_hR_max", type=float, default=8.0)
    p.add_argument("--slice_eval_nhR", type=int, default=80)
    p.add_argument("--slice_eval_fields", type=str, default="h,q,state", help="Comma-separated fields for slice eval.")

    args = p.parse_args()
    ROOT_CWD = os.getcwd()
    args = apply_config_json(args, p, ROOT_CWD)




    # Override default eval regimes to match ROM paper figures (Figures 5–8)
    _apply_eval_paper_figures_overrides(args)

    if args.sanity:
        args.num_epochs = int(args.sanity_epochs)
        args.Nx = 52
        # Give shocks time to propagate in sanity (still small Nx and coarse sampling -> fast)
        args.t_final = 1.0
        args.train_sample_interval = 0.1
        args.val_sample_interval = 0.1
        args.N_colloc_train = 600
        args.N_colloc_val = 150
        args.paper_eval_n_sample = 5
        args.test_t = [0.1, 0.5, 1.0]

        args.display = False
        args.disable_scheduler = True
        args.residual_scaling = "ema_rms"


        # Sanity should be fast, but must respect CLI shock/gating parameters.
        # Only force shock-aware on if the user didn't request otherwise.
        if not args.shock_aware_colloc:
            args.shock_aware_colloc = True
        if args.shock_indicator is None:
            args.shock_indicator = "h"

        # Include one dry/near-dry case and one non-dry case so sanity tests both
        # uniform dry-bed sampling and shock-aware focus/exclusion sampling.
        args.train_regime = [(10.0, 0.0, "perpendicular"), (26.0, 7.0, "perpendicular")]
        args.val_regime = (17.0, 4.0, "perpendicular")
        args.test_h = [(26.0, 0.14, "perpendicular")]
        if args.eval_paper_figures:
            args.test_h = [
                (12.0, 0.0, "perpendicular"),
                (15.0, 4.0, "perpendicular"),
                (26.0, 0.14, "perpendicular"),
            ]

        args.paper_eval = True
        args.paper_eval_sampling = "grid"
        args.paper_eval_exclude_H1 = True
        args.paper_eval_grid_hL = [10.0, 26.0]
        args.paper_eval_grid_hR = [0.0, 0.14, 8.0]

        args.paper_eval_Nx = args.Nx
        args.paper_eval_t_final = args.t_final
        args.paper_eval_dt = args.dt
        args.paper_eval_sample_interval = args.train_sample_interval

        # Enable slice_eval with a tiny sweep (5 points)
        args.slice_eval = True
        args.slice_eval_hL = 25.0
        args.slice_eval_hR_min = 0.0
        args.slice_eval_hR_max = 8.0
        args.slice_eval_nhR = 5
        args.slice_eval_fields = "h,q"

    _apply_train_grid_json(args, ROOT_CWD)
    _apply_paper_eval_grid_json(args, ROOT_CWD)

    # --- Greedy / multi-run mode ---
    if args.greedy_search:
        if args.greedy_json is None:
            raise ValueError("--greedy_search requires --greedy_json")

        with open(args.greedy_json, "r") as f:
            cfg = json.load(f)

        baseline_overrides = cfg.get("baseline", {})
        experiments = cfg.get("experiments", [])
        if not isinstance(experiments, list) or len(experiments) == 0:
            raise ValueError("greedy_json must contain a non-empty 'experiments' list")

        # Optionally prepend a baseline run (baseline overrides only)
        if getattr(args, "greedy_run_baseline", True):
            has_baseline_named = any((isinstance(e, dict) and str(e.get("name", "")).strip().lower() == "baseline") for e in experiments)
            if not has_baseline_named:
                experiments = [{"name": "baseline", "overrides": {}}] + experiments

        sweep_name = args.greedy_name
        if sweep_name is None or str(sweep_name).strip() == "":
            sweep_name = f"greedy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Desired layout: <output_dir>/experiments/<greedy_ts>/<exp_name>/experiments/<run_ts>/...
        # where greedy_ts is the greedy sweep name (default greedy_YYYYMMDD_HHMMSS).
        sweep_root = os.path.join(args.output_dir, "experiments", sweep_name)
        _safe_mkdir(sweep_root)

        rows = []
        max_runs = int(args.greedy_max_runs) if args.greedy_max_runs is not None else None
        # If baseline is prepended, greedy_max_runs applies to *non-baseline* experiments.
        baseline_name_lc = "baseline"

        for i, exp in enumerate(experiments):
            if max_runs is not None:
                # Count only non-baseline experiments toward the cap
                non_base_count = sum(1 for e in experiments[:i] if str(e.get("name", "")).strip().lower() != baseline_name_lc)
                if non_base_count >= max_runs and str(exp.get("name", "")).strip().lower() != baseline_name_lc:
                    break

            exp_name = exp.get("name", None)
            if exp_name is None or str(exp_name).strip() == "":
                exp_name = f"exp_{i:03d}"
            exp_name = str(exp_name)

            overrides = exp.get("overrides", {})
            # build a run-specific args copy
            run_args = _ns_copy(args)

            # apply baseline overrides then per-experiment overrides
            base_dict = vars(run_args)
            _deep_update(base_dict, baseline_overrides)
            _deep_update(base_dict, overrides)

            # Apply any grid JSONs introduced via baseline/experiment overrides.
            _apply_train_grid_json(run_args, ROOT_CWD)
            _apply_paper_eval_grid_json(run_args, ROOT_CWD)

            # If eval_paper_figures is enabled via CLI or JSON overrides, ensure test_h is updated.
            _apply_eval_paper_figures_overrides(run_args)

            # run in its own output_dir so artifacts are grouped by experiment name
            run_args.output_dir = os.path.join(sweep_root, exp_name)
            _safe_mkdir(run_args.output_dir)

            # Run full pipeline
            print(f"[greedy] Running {i+1}/{len(experiments)}: {exp_name}")
            ckpt_folder = _run_single_pipeline(run_args, ROOT_CWD)

            # Summarize metrics
            latest_exp = ckpt_folder if (ckpt_folder and os.path.isdir(ckpt_folder)) else _find_latest_experiment_folder(run_args.output_dir)
            metrics = _load_metrics_from_folder(latest_exp)

            row = {
                "exp_name": exp_name,
                "output_dir": run_args.output_dir,
                "checkpoint_folder": latest_exp,
                "sweep_root": sweep_root,
            }

            # keep a compact record of overrides
            try:
                row["overrides_json"] = json.dumps(overrides, sort_keys=True)
            except Exception:
                row["overrides_json"] = str(overrides)

            # add some key knobs for quick filtering
            for k in [
                "pde_weight", "data_weight", "bc_weight", "ic_weight", "lambda_q",
                "shock_aware_colloc", "shock_w_excl", "shock_w_focus", "shock_focus_share",
                "pde_gate", "gate_alpha_grad", "gate_h_min", "gate_floor",
                "residual_scaling", "residual_ema_beta",
                "Nx", "t_final", "dt", "N_colloc_train", "N_colloc_val",
                "num_epochs", "optimizer_lr",
            ]:
                if hasattr(run_args, k):
                    row[k] = getattr(run_args, k)

            # merge in metrics
            row.update(metrics)
            rows.append(row)

            # write partial CSV after each run
            _write_greedy_csv(rows, os.path.join(sweep_root, "greedy_summary.csv"))

        print(f"[greedy] Done. Wrote {os.path.join(sweep_root, 'greedy_summary.csv')}")
        return

    # --- Standard single-run mode ---
    _run_single_pipeline(args, ROOT_CWD)
    return

if __name__ == "__main__":
    main()
