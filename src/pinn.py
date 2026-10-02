import os, time, datetime, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, torch.autograd as autograd
import matplotlib
matplotlib.use('Agg') # switch to a non-interactive backend
import matplotlib.pyplot as plt

from typing import Any, Callable
from data_generator import simulate_shallow_water
from architectures import Lin_Architecture, ResidualLin_Architecture
import plotting_utils
from experiment_utils import TimingRecorder, dump_json


class Linear_PINN(nn.Module):
    def __init__(
        self,
        architecture: dict[str, Any] | None = None,
        activation: nn.Module | None = None,
        num_epochs: int = 12000,
        optimizer: Callable[..., optim.Optimizer] = optim.Adam,
        optimizer_lr: float = 1e-3,
        data_weight: float = 1.0,
        pde_weight: float = 0.1,
        l2: dict[str, bool] | None = None,
        bc_weight: float = 1.0,
        ic_weight: float = 1.0,
        nonneg_h_weight: float = 0.0,
        lambda_q: float = 1.0,
        enforce_pos_h: str = "none",   # "none" or "softplus"
        rampup_epochs_share: float = 0.5,
        scheduler_params: dict[str, Any] | None = None,
        grad_clip: float = 1.0,
        display: bool = False,
        device: str = "legacy_auto",
    ) -> None:
        """Initialize the PINN model and training configuration."""
        super(Linear_PINN, self).__init__()
        architecture = architecture or {'model': 'lin', 'layers': [4, 120, 120, 120, 120, 120, 2]}
        activation = activation if activation is not None else nn.SiLU()
        l2 = l2 or {'data': False, 'pde': False}
        scheduler_params = scheduler_params or {"status": True, "T_max": 12000, "eta_min": 1e-5}
        self.timings = TimingRecorder()
        layers = architecture['layers']
        model_name = architecture['model']
        if model_name == 'lin':
            self.model = Lin_Architecture(layers=layers, activation=activation)
        elif model_name == 'residual_lin':
            self.model = ResidualLin_Architecture(layers=layers, activation=activation)
        else:
            raise ValueError(f"Unsupported public architecture: {model_name}")
            
        self.optimizer = optimizer(self.model.parameters(), lr=optimizer_lr)
        self.g_const = 9.81
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.ic_weight = ic_weight
        self.l2 = l2
        self.nonneg_h_weight = float(nonneg_h_weight)
        self.lambda_q = float(lambda_q)
        self.enforce_pos_h = str(enforce_pos_h).lower()
        if self.enforce_pos_h not in {"none", "softplus"}:
            raise ValueError(f"enforce_pos_h must be 'none' or 'softplus', got: {self.enforce_pos_h}")
        self.rampup_epochs_share = rampup_epochs_share
        self.grad_clip = grad_clip
        self.num_epochs = num_epochs
        device = str(device).lower()
        if device == "legacy_auto":
            # Preserve the submitted implementation exactly: CUDA when available, CPU otherwise.
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but CUDA is unavailable")
            selected_device = "cuda"
        elif device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("--device mps requested but Apple MPS is unavailable")
            selected_device = "mps"
        elif device == "cpu":
            selected_device = "cpu"
        else:
            raise ValueError(f"Unsupported device: {device}")
        self.device = torch.device(selected_device)
        self.model.to(self.device)
        self.display = display
        if scheduler_params['status']:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                        T_max=scheduler_params['T_max'], 
                                                                        eta_min=scheduler_params['eta_min'])
        else:
            self.scheduler = None
        self.checkpoint_folder = os.path.join("experiments", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.get_data_params = {}
        self.norm = {"mu_h":0., "std_h":1., "mu_u":0., "std_u":1., "lambda_u":float(lambda_q)}
        print(f"PINN: {model_params:,} trainable parameters; device={self.device}")

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns model outputs for input features."""
        return self.model(x)

    
    
    def _split_outputs(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split outputs into (h,u) and optionally enforce h>0 via softplus."""
        h_raw = output[:, 0:1]
        u = output[:, 1:2]
        if getattr(self, "enforce_pos_h", "none") == "softplus":
            h = F.softplus(h_raw) + 1e-6
        else:
            h = h_raw
        return h, u
    
    @staticmethod
    def _q_from_hu_np(h: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.asarray(h) * np.asarray(u)

    @staticmethod
    def _q_from_hu_torch(h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return h * u

    # ---- parameterization & robust penalty helpers --------------------------------
    @staticmethod
    def _to_invariants(p_np: np.ndarray) -> np.ndarray:
        """map [[hL,hR], ...] -> [[h_avg, gamma], ...] in numpy"""
        hL = p_np[:, 0]; hR = p_np[:, 1]
        h_avg = 0.5 * (hL + hR)
        gamma  = (hL - hR) / np.maximum(2.0*h_avg, 1e-6)
        return np.stack([h_avg, gamma], axis=1)

    @staticmethod
    def _to_invariants_torch(p_t: torch.Tensor) -> torch.Tensor:
        """map [[hL,hR], ...] -> [[h_avg, gamma], ...] in torch"""
        hL, hR = p_t[:, 0:1], p_t[:, 1:2]
        h_avg = 0.5 * (hL + hR)
        gamma = (hL - hR) / torch.clamp(2.0*h_avg, min=1e-6)
        return torch.cat([h_avg, gamma], dim=1)

    @staticmethod
    def _rho(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Charbonnier penalty (robust L1)"""
        return torch.sqrt(x*x + eps*eps)

    def _normalize_pde_residuals(
        self,
        r1: torch.Tensor,
        r2: torch.Tensor,
        cfg: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Normalize PDE residual components to improve balancing.

        Modes (cfg['residual_scaling']):
          - 'none'     : no scaling
          - 'batch_rms': divide each residual by its batch RMS (detached)
          - 'ema_rms'  : divide by EMA of RMS (detached), more stable than per-batch

        Returns normalized (r1n, r2n) and a stats dict with the current scales.
        """
        cfg = cfg or {}
        mode = str(cfg.get('residual_scaling', 'none')).lower()
        eps = float(cfg.get('residual_scale_eps', 1e-8))

        if mode == 'none':
            return r1, r2, {'s1': 1.0, 's2': 1.0}

        # batch RMS (detach so the model cannot game the scale)
        s1_batch = torch.sqrt(torch.mean(r1.detach() ** 2) + eps)
        s2_batch = torch.sqrt(torch.mean(r2.detach() ** 2) + eps)

        if mode == 'batch_rms':
            r1n = r1 / s1_batch
            r2n = r2 / s2_batch
            return r1n, r2n, {'s1': float(s1_batch.item()), 's2': float(s2_batch.item())}

        if mode == 'ema_rms':
            beta = float(cfg.get('residual_ema_beta', 0.99))

            # lazily initialize EMA state
            if not hasattr(self, '_res_scale_ema'):
                self._res_scale_ema = {
                    's1': float(s1_batch.item()),
                    's2': float(s2_batch.item()),
                }
            else:
                self._res_scale_ema['s1'] = beta * self._res_scale_ema['s1'] + (1.0 - beta) * float(s1_batch.item())
                self._res_scale_ema['s2'] = beta * self._res_scale_ema['s2'] + (1.0 - beta) * float(s2_batch.item())

            s1 = torch.tensor(self._res_scale_ema['s1'], device=r1.device, dtype=r1.dtype)
            s2 = torch.tensor(self._res_scale_ema['s2'], device=r2.device, dtype=r2.dtype)
            r1n = r1 / s1
            r2n = r2 / s2
            return r1n, r2n, {'s1': float(self._res_scale_ema['s1']), 's2': float(self._res_scale_ema['s2'])}

        # fallback
        return r1, r2, {'s1': 1.0, 's2': 1.0}
    
    def _pde_gating_weights(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        param: torch.Tensor,
        cfg: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Multiplicative weights for PDE residuals.

        Downweights PDE residuals:
        - near steep gradients (proxy: |h_x|)
        - near wet/dry front where h is very small

        Indicators are detached so the network cannot "cheat" by shaping the gate.
        Returns shape (N,1).
        """
        cfg = cfg or {}
        if not bool(cfg.get('pde_gate', False)):
            return torch.ones_like(x)

        alpha = float(cfg.get('gate_alpha_grad', 8.0))
        h_min = float(cfg.get('gate_h_min', 0.05))
        floor = float(cfg.get('gate_floor', 0.05))

        # Forward pass for h (needs grad wrt x to estimate |h_x|)
        inp = torch.cat([x, t, param], dim=1)

        out = self.model(inp)
        h, _ = self._split_outputs(out)
        h_x = autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True, retain_graph=True)[0]

        # Indicators (detach so the gate is not optimized directly)
        g = torch.abs(h_x).detach()
        h_det = h.detach()

        # Gradient gate
        w_grad = torch.exp(-alpha * g)

        # Dry-front gate (stable clipped linear ramp)
        w_h = torch.clamp((h_det - h_min) / max(h_min, 1e-6), min=0.0, max=1.0)

        w = torch.clamp(w_grad * w_h, min=floor, max=1.0)
        return w

    # ---- shock localization helpers --------------------------------------------
    @staticmethod
    def _estimate_shock_x_from_snapshot(
        snap: dict[str, Any],
        indicator: str = "h",
        smooth_window: int = 7,
        boundary_buf_frac: float = 0.02,
    ) -> float:
        """Estimate dominant shock/front location x(t) from a simulator snapshot.

        Improvements:
        - Light smoothing to reduce jumps between nearby steep features.
        - Ignores a small buffer near boundaries to avoid boundary artifacts.
        - For strong-form PINNs, the problematic region is the steepest jump.
        """
        x = np.asarray(snap["x"]).reshape(-1)
        h = np.asarray(snap["h"]).reshape(-1)

        if indicator.lower() == "u":
            y = np.asarray(snap["u"]).reshape(-1)
        else:
            y = h

        # moving-average smoothing
        w = int(max(1, smooth_window))
        if w >= 3:
            if (w % 2) == 0:
                w += 1
            kernel = np.ones(w, dtype=float) / float(w)
            y_s = np.convolve(y, kernel, mode="same")
        else:
            y_s = y

        dx = float(x[1] - x[0])
        g = np.empty_like(y_s)
        g[1:-1] = (y_s[2:] - y_s[:-2]) / (2.0 * dx)
        g[0] = (y_s[1] - y_s[0]) / dx
        g[-1] = (y_s[-1] - y_s[-2]) / dx

        # ignore boundary buffer (FV can have boundary artifacts / ghost-cell effects)
        n = len(x)
        frac = boundary_buf_frac
        buf = max(5, int(frac * n))                  # enforce minimum cells too
        i0 = buf
        i1 = n - buf
        if i1 <= i0 + 2:
            i0 = 2
            i1 = n - 2

        gg = g[i0:i1]
        
        idx_local = int(np.argmax(np.abs(gg)))

        idx = idx_local + i0
        return float(x[idx])

    def _build_shock_trajectories(self, snapshots: list[dict[str, Any]], indicator: str = 'h', include_t0: bool = False) -> None:
        """Build per-regime shock trajectories x_shock(t) from a list of simulator snapshots.

        Stores:
            self.shock_traj[(hL,hR,shape)] = {"t": np.ndarray, "x": np.ndarray}

        Notes
        -----
        - This uses the supervised snapshots (truth) you already generated, so it is cheap.
        - We keep it lightweight: a single dominant front per snapshot (can be extended later).
        """
        traj: dict[tuple[float, float, str], dict[str, list[float]]] = {}
        for snap in snapshots:
            if (not include_t0) and abs(float(snap.get('t', 0.0))) < 1e-12:
                continue

            p = snap.get('param', None)
            if p is None:
                continue
            hL = float(p[0]); hR = float(p[1])
            shape = str(snap.get('dam_shape', 'perpendicular'))
            key = (hL, hR, shape)
            if key not in traj:
                traj[key] = {"t": [], "x": []}
            traj[key]["t"].append(float(snap['t']))
            cfg = getattr(self, "shock_cfg_for_traj", {}) or {}
            sw = int(cfg.get("shock_smooth_window", 7))
            bf = float(cfg.get("shock_boundary_buf_frac", 0.02))
            traj[key]["x"].append(
                self._estimate_shock_x_from_snapshot(
                    snap, indicator=indicator, smooth_window=sw, boundary_buf_frac=bf
                )
            )

        # sort each trajectory by time
        self.shock_traj = {}
        for key, d in traj.items():
            t_arr = np.asarray(d["t"], dtype=float)
            x_arr = np.asarray(d["x"], dtype=float)
            order = np.argsort(t_arr)
            self.shock_traj[key] = {"t": t_arr[order], "x": x_arr[order]}


    def _shock_x_at(self, key: tuple[float, float, str], t: float) -> float:
        """Interpolate x_shock(t) for a regime key = (hL,hR,shape)."""
        if not hasattr(self, 'shock_traj'):
            raise RuntimeError("shock_traj not built; call _build_shock_trajectories first")
        if key not in self.shock_traj:
            # fallback: center
            return 0.5 * float(self.Lx)
        tt = self.shock_traj[key]["t"]
        xx = self.shock_traj[key]["x"]
        if len(tt) == 0:
            return 0.5 * float(self.Lx)
        # clamp
        if t <= tt[0]:
            return float(xx[0])
        if t >= tt[-1]:
            return float(xx[-1])
        return float(np.interp(t, tt, xx))


    def _sample_x_shock_aware(self, xS: float, Lx: float, n: int, w_excl: float, w_focus: float, focus_share: float) -> np.ndarray:
        """Sample x values in [0,Lx] avoiding |x-xS|<w_excl and optionally oversampling near shock.

        Returns array shape (n,1).
        """
        xs = np.empty((n, 1), dtype=float)
        for i in range(n):
            r = np.random.rand()
            if r < focus_share:
                # near-shock band excluding the inner exclusion band
                left_a = max(0.0, xS - w_focus)
                left_b = max(0.0, xS - w_excl)
                right_a = min(Lx, xS + w_excl)
                right_b = min(Lx, xS + w_focus)
                # choose left vs right proportional to length
                L1 = max(0.0, left_b - left_a)
                L2 = max(0.0, right_b - right_a)
                if (L1 + L2) <= 1e-12:
                    # fallback to uniform outside exclusion
                    r2 = np.random.rand()
                else:
                    r2 = np.random.rand() * (L1 + L2)
                if r2 < L1 and L1 > 0:
                    xs[i, 0] = np.random.uniform(left_a, left_b)
                elif L2 > 0:
                    xs[i, 0] = np.random.uniform(right_a, right_b)
                else:
                    # fallback
                    xs[i, 0] = np.random.uniform(0.0, Lx)
            else:
                # uniform away from the exclusion band
                left_len = max(0.0, (xS - w_excl) - 0.0)
                right_len = max(0.0, Lx - (xS + w_excl))
                tot = left_len + right_len
                if tot <= 1e-12:
                    xs[i, 0] = np.random.uniform(0.0, Lx)
                else:
                    r2 = np.random.rand() * tot
                    if r2 < left_len:
                        xs[i, 0] = np.random.uniform(0.0, xS - w_excl)
                    else:
                        xs[i, 0] = np.random.uniform(xS + w_excl, Lx)
        return xs
    
    def shock_sampling_sanity_plot(
        self,
        regimes=None,
        times=None,
        n_per_time: int = 2000,
        save_name: str = "shock_sampling_sanity.png",
    ):
        """Save a quick histogram sanity check for shock-aware collocation sampling."""
        return plotting_utils.shock_sampling_sanity_plot(
            self,
            regimes=regimes,
            times=times,
            n_per_time=n_per_time,
            save_name=save_name,
        )
    

    def get_data(
        self,
        train_regimes: list[tuple[float, float, str]] | None = None,
        train_sample_interval: float = 0.1,
        val_regime: tuple[float, float, str] = (12.0, 0.0, 'perpendicular'),
        val_sample_interval: float = 0.2,
        Lx: float = 100.0,
        Nx: int = 1002,
        t_final: float = 2.0,
        dt: float = 1e-3,
        tol: float | None = None, #1e-1,
        N_colloc_train: int | None = None, #20000,
        N_colloc_val: int | None = None, #4000,
        shock_aware_colloc: dict[str, Any] | None = None,
        include_t0: bool = False
    ) -> None:
        """Generate simulator data and build tensors for supervised and collocation training/validation."""
        train_regimes = train_regimes or [(5.0, 0.0, 'perpendicular'), (10.0, 0.0, 'perpendicular'), (15.0, 0.0, 'perpendicular')]
        default_shock_cfg = {
            'status': False, 'indicator': 'h', 'w_excl': 1.5, 'w_focus': 6.0, 'focus_share': 0.6,
            'dry_hR_eps': 0.5, 'pde_gate': True, 'gate_alpha_grad': 8.0, 'gate_h_min': 0.05,
            'gate_floor': 0.05, 'shock_smooth_window': 7, 'shock_boundary_buf_frac': 0.02,
            'residual_scaling': 'none', 'residual_ema_beta': 0.99, 'residual_scale_eps': 1e-8,
            'balance_dry_loss': False, 'dry_loss_weight': 3.0,
        }
        if shock_aware_colloc:
            default_shock_cfg.update(shock_aware_colloc)
        shock_aware_colloc = default_shock_cfg
        data_total_start = time.perf_counter()
        dx = Lx/(Nx - 2)
        if tol is None or tol <= 0:
            tol = 0.6 * dx  # safely grabs 1st/last interior points
        self.train_regimes = train_regimes
        self.t_final = t_final
        self.Lx = Lx
        simulated_data = []
        _train_fom_start = time.perf_counter()
        for regime in self.train_regimes:
            h_left, h_right, dam_shape = regime 
            snapshots = simulate_shallow_water(h_left=h_left,
                                               h_right=h_right,
                                               dam_shape=dam_shape,
                                               Lx=Lx,
                                               Nx=Nx,
                                               t_final=self.t_final, 
                                               dt=dt,
                                               sample_interval=train_sample_interval,
                                               plot=False)
            simulated_data.extend(snapshots)
        self.timings.add('fom_training_dataset_generation', time.perf_counter() - _train_fom_start)


        X_train, T_train, Param_train, h_train, u_train = [], [], [], [], []
        
        # organize the data into arrays; note that "param" is now a 2-element vector
        for snap in simulated_data:
            if (not include_t0) and (abs(snap['t']) < 1e-12):
                continue
            n_points = len(snap['x'])
            idx = np.arange(n_points)

            X_train.append(snap['x'][idx, None])
            T_train.append(np.full((len(idx), 1), snap['t']))
            Param_train.append(np.tile(snap['param'], (len(idx), 1)))
            h_train.append(snap['h'][idx, None])
            u_train.append(snap['u'][idx, None])
        
        X_train = np.vstack(X_train)
        T_train = np.vstack(T_train)
        # Param_train = np.vstack(Param_train)
        P = np.vstack(Param_train)  # columns: [hL, hR]
        # strict dry-bed mask: ONLY hR == 0 (no epsilon-based wet/dry classification)
        is_dry_train = (np.abs(P[:, 1]) <= 1e-12).astype(np.float32).reshape(-1, 1)
        h_avg = 0.5*(P[:,0]+P[:,1])
        gamma = (P[:,0]-P[:,1]) / np.maximum(h_avg*2.0, 1e-6)
        Param_train = np.stack([h_avg, gamma], axis=1)
        h_train = np.vstack(h_train)
        u_train = np.vstack(u_train)
        
        # ----- standardization stats (used only inside loss) -----
        self.norm.update({
            "mu_h": float(np.mean(h_train)),
            "std_h": float(np.std(h_train) + 1e-6),
            "mu_u": float(np.mean(u_train)),
            "std_u": float(np.std(u_train) + 1e-6),
        })

        print(f"Generated {X_train.shape[0]} training data points")

        # Extract boundary points for training (assumes domain [0, Lx])
        boundary_mask_train = ((X_train <= tol) | (X_train >= (Lx - tol))).flatten()
        if sum(boundary_mask_train)==0:
            return 'ERROR: tol is not proper to find boundary points'
            
        bc_train = {
            "x": torch.tensor(X_train[boundary_mask_train].reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=True),
            "t": torch.tensor(T_train[boundary_mask_train].reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=True),
            "param": torch.tensor(Param_train[boundary_mask_train].reshape(-1, 2), dtype=torch.float32, device=self.device, requires_grad=True),
        }

        # Analytic initial-condition points for training.
        # For the 1D dam-break setup, h(x,0)=hL for x<Lx/2, hR for x>=Lx/2, and u(x,0)=0.
        X_ic_train, T_ic_train, Param_ic_train, h_ic_train, u_ic_train = [], [], [], [], []
        for h_left, h_right, dam_shape in train_regimes:
            x_ic = simulated_data[0]["x"].reshape(-1, 1)
            t_ic = np.zeros_like(x_ic)
            p_ic_raw = np.tile(np.array([float(h_left), float(h_right)]), (x_ic.shape[0], 1))
            p_ic = self._to_invariants(p_ic_raw)
            h_ic = np.where(x_ic < 0.5 * Lx, float(h_left), float(h_right))
            u_ic = np.zeros_like(x_ic)

            X_ic_train.append(x_ic)
            T_ic_train.append(t_ic)
            Param_ic_train.append(p_ic)
            h_ic_train.append(h_ic)
            u_ic_train.append(u_ic)

        X_ic_train = np.vstack(X_ic_train)
        T_ic_train = np.vstack(T_ic_train)
        Param_ic_train = np.vstack(Param_ic_train)
        h_ic_train = np.vstack(h_ic_train)
        u_ic_train = np.vstack(u_ic_train)
        P_ic_train_raw = np.vstack([
            np.tile(np.array([float(h_left), float(h_right)]), (simulated_data[0]["x"].reshape(-1, 1).shape[0], 1))
            for h_left, h_right, _ in train_regimes
        ])
        is_dry_ic_train = (np.abs(P_ic_train_raw[:, 1]) <= 1e-12).astype(np.float32).reshape(-1, 1)

        ic_train = {
            "x": torch.tensor(X_ic_train, dtype=torch.float32, device=self.device, requires_grad=True),
            "t": torch.tensor(T_ic_train, dtype=torch.float32, device=self.device, requires_grad=True),
            "param": torch.tensor(Param_ic_train, dtype=torch.float32, device=self.device, requires_grad=True),
            "h": torch.tensor(h_ic_train, dtype=torch.float32, device=self.device),
            "u": torch.tensor(u_ic_train, dtype=torch.float32, device=self.device),
            "is_dry": torch.tensor(is_dry_ic_train, dtype=torch.float32, device=self.device),
        }

        # validation FOM trajectory
        _val_fom_start = time.perf_counter()
        val_snapshots = simulate_shallow_water(h_left=val_regime[0],
                                               h_right=val_regime[1],
                                               dam_shape=val_regime[2],
                                               Lx=Lx,
                                               Nx=Nx,
                                               t_final=self.t_final, 
                                               dt=dt,
                                               sample_interval=val_sample_interval,
                                               plot=False)
        self.timings.add('fom_validation_generation', time.perf_counter() - _val_fom_start)
        _shock_indicator = str(shock_aware_colloc.get('indicator', 'h'))
        self.shock_cfg_for_traj = dict(shock_aware_colloc)
        _shock_pre_start = time.perf_counter()
        self._build_shock_trajectories(list(simulated_data), indicator=_shock_indicator, include_t0=False)
        self.timings.add('shock_detection_preprocessing', time.perf_counter() - _shock_pre_start)

        X_val, T_val, Param_val, h_val, u_val = [], [], [], [], []
        for snap in val_snapshots:
            n_points = len(snap['x'])
            X_val.append(snap['x'].reshape(n_points, 1))
            T_val.append(np.full((n_points, 1), snap['t']))
            Param_val.append(np.tile(snap['param'], (n_points, 1)))
            h_val.append(snap['h'].reshape(n_points, 1))
            u_val.append(snap['u'].reshape(n_points, 1))
        
        X_val = np.vstack(X_val)
        T_val = np.vstack(T_val)
        # Param_val = np.vstack(Param_val)
        P = np.vstack(Param_val)  # columns: [hL, hR]
        is_dry_val = (np.abs(P[:, 1]) <= 1e-12).astype(np.float32).reshape(-1, 1)
        h_avg = 0.5*(P[:,0]+P[:,1])
        gamma = (P[:,0]-P[:,1]) / np.maximum(h_avg*2.0, 1e-6)
        Param_val = np.stack([h_avg, gamma], axis=1)
        h_val = np.vstack(h_val)
        u_val = np.vstack(u_val)
        
        print(f"Generated {X_val.shape[0]} validation data points")

        # Extract boundary points for validation
        boundary_mask_val = ((X_val <= tol) | (X_val >= (Lx - tol))).flatten()
        bc_val = {
            "x": torch.tensor(X_val[boundary_mask_val].reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=True),
            "t": torch.tensor(T_val[boundary_mask_val].reshape(-1, 1), dtype=torch.float32, device=self.device, requires_grad=True),
            "param": torch.tensor(Param_val[boundary_mask_val].reshape(-1, 2), dtype=torch.float32, device=self.device, requires_grad=True),
        }

        # Analytic initial-condition points for validation regime.
        x_ic_val = val_snapshots[0]["x"].reshape(-1, 1)
        t_ic_val = np.zeros_like(x_ic_val)
        p_ic_val_raw = np.tile(np.array([float(val_regime[0]), float(val_regime[1])]), (x_ic_val.shape[0], 1))
        p_ic_val = self._to_invariants(p_ic_val_raw)
        h_ic_val = np.where(x_ic_val < 0.5 * Lx, float(val_regime[0]), float(val_regime[1]))
        u_ic_val = np.zeros_like(x_ic_val)
        is_dry_ic_val = (np.abs(p_ic_val_raw[:, 1]) <= 1e-12).astype(np.float32).reshape(-1, 1)

        ic_val = {
            "x": torch.tensor(x_ic_val, dtype=torch.float32, device=self.device, requires_grad=True),
            "t": torch.tensor(t_ic_val, dtype=torch.float32, device=self.device, requires_grad=True),
            "param": torch.tensor(p_ic_val, dtype=torch.float32, device=self.device, requires_grad=True),
            "h": torch.tensor(h_ic_val, dtype=torch.float32, device=self.device),
            "u": torch.tensor(u_ic_val, dtype=torch.float32, device=self.device),
            "is_dry": torch.tensor(is_dry_ic_val, dtype=torch.float32, device=self.device),
        }

        # Store full training and validation data as tensors
        self.data = {
            "train": {
                "x": torch.tensor(X_train, dtype=torch.float32, device=self.device, requires_grad=True),
                "t": torch.tensor(T_train, dtype=torch.float32, device=self.device, requires_grad=True),
                "param": torch.tensor(Param_train, dtype=torch.float32, device=self.device, requires_grad=True),
                "h": torch.tensor(h_train, dtype=torch.float32, device=self.device),
                "u": torch.tensor(u_train, dtype=torch.float32, device=self.device),
                "is_dry": torch.tensor(is_dry_train, dtype=torch.float32, device=self.device),
                "bc": bc_train,
                "ic": ic_train
            },
            "val": {
                "x": torch.tensor(X_val, dtype=torch.float32, device=self.device, requires_grad=True),
                "t": torch.tensor(T_val, dtype=torch.float32, device=self.device, requires_grad=True),
                "param": torch.tensor(Param_val, dtype=torch.float32, device=self.device, requires_grad=True),
                "h": torch.tensor(h_val, dtype=torch.float32, device=self.device),
                "u": torch.tensor(u_val, dtype=torch.float32, device=self.device),
                "is_dry": torch.tensor(is_dry_val, dtype=torch.float32, device=self.device),
                "bc": bc_val,
                "ic": ic_val
            }
        }

        # collocation points for PDE residual (sampled uniformly; fewer points for validation)
        # x in [0, Lx], t in [0, t_final], and param in [min_param, max_param]
        h_left_vals = [regime[0] for regime in train_regimes]
        h_right_vals = [regime[1] for regime in train_regimes]
        h_left_min, h_left_max = min(h_left_vals), max(h_left_vals)
        h_right_min, h_right_max = min(h_right_vals), max(h_right_vals)
        # Parameter hull from the training regimes
        hL_rng = max(h_left_max - h_left_min, 0.0)
        hR_rng = max(h_right_max - h_right_min, 0.0)
        n_hL_unique = len(sorted(set(h_left_vals)))
        n_hR_unique = len(sorted(set(h_right_vals)))

        # Auto-size collocation counts if the user did not specify them
        if (N_colloc_train is None) or (N_colloc_val is None):
            # Target resolutions:
            # In x: collocation spacing ~ kx * dx (coarser than the FOM grid by factor kx >= 1)
            # In t: aim for Nt collocation “slices” across [0, T]
            # In params: aim for BhL and BhR bins across each range (or use counting measure if range is 0)
            kx = 3.0 # target spatial subsampling factor (dx_* = kx * dx)
            Nt = max(10, int(np.ceil(t_final / max(train_sample_interval, 1e-8)))) # collocate at least once per saved snapshot
            BhL = max(3, n_hL_unique) # target bins along h_L (>= number of unique training h_L)
            BhR = max(5, n_hR_unique) # target bins along h_R (>= number of unique training h_R)

            # Compute axis-wise counts implied by the targets
            n_x = int(np.ceil(Lx / (kx * dx)))
            n_x = max(2, n_x)  # at least 2 points in x
            n_t = int(np.ceil(Nt))
            n_t = max(2, n_t)  # at least 2 times
            n_hL = int(np.ceil(BhL)) if hL_rng > 0 else max(1, n_hL_unique) # equivalently dhL_* = hL_rng / BhL
            n_hR = int(np.ceil(BhR)) if hR_rng > 0 else max(1, n_hR_unique) # equivalently dhR_* = hR_rng / BhR

            # Total counts
            N_train_auto = int(n_x * n_t * n_hL * n_hR)
            val_to_train_ratio = 0.1
            N_val_auto = int(max(1, np.round(val_to_train_ratio * N_train_auto)))

            # If only one of the two was None, preserve the user-specified one
            if N_colloc_train is None:
                N_colloc_train = N_train_auto
            if N_colloc_val is None:
                N_colloc_val = N_val_auto

        # ---------------- collocation points for PDE residual -----------------
        # NOTE: For this dam-break problem, pointwise (strong-form) residuals at the discontinuity
        # can harm training. We therefore support a regime/time-dependent shock-aware sampler that:
        #   (1) excludes a narrow band around x_shock(t)
        #   (2) oversamples just outside that band.

        shock_status = bool(shock_aware_colloc.get('status', False))
        _colloc_start = time.perf_counter()

        # ---- colloc (train) ----
        if shock_status:
            w_excl = float(shock_aware_colloc.get('w_excl', 1.5))
            w_focus = float(shock_aware_colloc.get('w_focus', 6.0))
            focus_share = float(shock_aware_colloc.get('focus_share', 0.6))
            dry_hR_eps = float(shock_aware_colloc.get('dry_hR_eps', 0.5))
            if not np.isfinite(w_excl):
                w_excl_eff = 0.0
            else:
                w_excl_eff = float(max(0.0, w_excl))

            x_min_center = 0.0
            x_max_center = float(Lx)

            keys = [(float(r[0]), float(r[1]), str(r[2])) for r in train_regimes]
            ridx = np.random.randint(0, len(keys), size=N_colloc_train)
            t_coll = np.random.uniform(0.0, t_final, (N_colloc_train, 1))

            x_coll = np.empty((N_colloc_train, 1), dtype=float)
            P_coll = np.empty((N_colloc_train, 2), dtype=float)  # (hL,hR)

            for i in range(N_colloc_train):
                key = keys[int(ridx[i])]
                hL, hR, shape = key
                P_coll[i, 0] = hL
                P_coll[i, 1] = hR
                if float(hR) < dry_hR_eps:
                    x_coll[i, 0] = np.random.uniform(x_min_center, x_max_center)
                else:
                    xS = self._shock_x_at(key, float(t_coll[i, 0]))
                    x_coll[i:i+1, :] = self._sample_x_shock_aware(
                        xS=xS, Lx=Lx, n=1, w_excl=w_excl_eff, w_focus=w_focus, focus_share=focus_share
                    )

            P_train_colloc = self._to_invariants(P_coll)

            self.colloc = {
                "train": {
                    "x": torch.tensor(x_coll, dtype=torch.float32, device=self.device, requires_grad=True),
                    "t": torch.tensor(t_coll, dtype=torch.float32, device=self.device, requires_grad=True),
                    "param": torch.tensor(P_train_colloc, dtype=torch.float32, device=self.device, requires_grad=True),
                },
                "val": {}
            }
        else:
            x_coll = np.random.uniform(0.0, Lx, (N_colloc_train, 1))
            t_coll = np.random.uniform(0.0, t_final, (N_colloc_train, 1))
            P_coll = np.hstack([
                np.random.uniform(h_left_min,  h_left_max,  (N_colloc_train, 1)),
                np.random.uniform(h_right_min, h_right_max, (N_colloc_train, 1)),
            ])
            P_train_colloc = self._to_invariants(P_coll)

            self.colloc = {
                "train": {
                    "x": torch.tensor(x_coll, dtype=torch.float32, device=self.device, requires_grad=True),
                    "t": torch.tensor(t_coll, dtype=torch.float32, device=self.device, requires_grad=True),
                    "param": torch.tensor(P_train_colloc, dtype=torch.float32, device=self.device, requires_grad=True),
                },
                "val": {}
            }

        # ---- colloc (val) ----
        # For validation collocation, we keep it simple and uniform in (x,t) and in the training parameter hull.
        t_val_coll = np.random.uniform(0.0, t_final, (N_colloc_val, 1))
        x_val_coll = np.random.uniform(0.0, Lx, (N_colloc_val, 1))
        P_val = np.hstack([
            np.random.uniform(h_left_min,  h_left_max,  (N_colloc_val, 1)),
            np.random.uniform(h_right_min, h_right_max, (N_colloc_val, 1)),
        ])
        P_val_colloc = self._to_invariants(P_val)

        self.colloc["val"] = {
            "x": torch.tensor(x_val_coll, dtype=torch.float32, device=self.device, requires_grad=True),
            "t": torch.tensor(t_val_coll, dtype=torch.float32, device=self.device, requires_grad=True),
            "param": torch.tensor(P_val_colloc, dtype=torch.float32, device=self.device, requires_grad=True)
        }
        self.timings.add('collocation_sampling', time.perf_counter() - _colloc_start)

        
        self.get_data_params = {
            "train_regimes": train_regimes,
            "train_sample_interval": train_sample_interval,
            "val_regime": val_regime,
            "val_sample_interval": val_sample_interval,
            "Lx": Lx,
            "Nx": Nx,
            "t_final": t_final,
            "dt": dt,
            "tol": tol,
            "N_colloc_train": N_colloc_train,
            "N_colloc_val": N_colloc_val,
            "N_datapoints_train": X_train.shape[0],
            "N_datapoints_val": X_val.shape[0],
            "N_ic_train": X_ic_train.shape[0],
            "N_ic_val": x_ic_val.shape[0],
            "shock_aware_colloc": shock_aware_colloc
        }
        print(self.get_data_params)
        self.timings.add('data_preparation_total', time.perf_counter() - data_total_start)
        self.timings.save(os.path.join(self.checkpoint_folder, 'timing_summary.json'))
        with open(os.path.join(self.checkpoint_folder, 'experiment_settings.txt'), 'a') as f:
            f.write("\n\n=== get_data PARAMETERS ===\n")
            f.write(str(self.get_data_params))
    def compute_ic_loss(self, ic_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Analytic initial-condition loss with the same standardization as the data term."""
        X_input_ic = torch.cat([ic_data["x"], ic_data["t"], ic_data["param"]], dim=1)
        pred_ic = self.model(X_input_ic)
        h_pred, u_pred = self._split_outputs(pred_ic)

        e_h = (h_pred - ic_data["h"]) / self.norm["std_h"]
        e_u = (u_pred - ic_data["u"]) / self.norm["std_u"]
        lam_u = float(self.norm.get("lambda_u", 1.0))

        shock_cfg = (self.get_data_params.get('shock_aware_colloc', {}) if hasattr(self, 'get_data_params') else {})
        if bool(shock_cfg.get('balance_dry_loss', False)) and ("is_dry" in ic_data):
            w_dry = float(shock_cfg.get('dry_loss_weight', 3.0))
            w = 1.0 + (w_dry - 1.0) * ic_data["is_dry"]
        else:
            w = 1.0

        if self.l2['data']:
            return torch.mean(w * (e_h**2)) + lam_u * torch.mean(w * (e_u**2))
        return torch.mean(w * torch.abs(e_h)) + lam_u * torch.mean(w * torch.abs(e_u))






    def pde_residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        param: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute strong-form PDE residuals (mass, momentum) at collocation points."""
        X_input = torch.cat([x, t, param], dim=1)
            
        output = self.model(X_input)
        h, u = self._split_outputs(output)

        h_safe = torch.clamp(h, min=1e-6)
        hu = h_safe * u
        
        # Compute time derivatives
        h_t = autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True, retain_graph=True)[0]
        # For the mass conservation term we need spatial derivative of q = hu
        hu_x = autograd.grad(hu, x, grad_outputs=torch.ones_like(hu), create_graph=True, retain_graph=True)[0]
        r1 = h_t + hu_x  # residual for continuity, Mass
        
        # For momentum: time derivative of q
        hu_t = autograd.grad(hu, t, grad_outputs=torch.ones_like(hu), create_graph=True, retain_graph=True)[0]
        flux_mom = (hu * u) + 0.5 * self.g_const * (h_safe**2)
        flux_mom_x = autograd.grad(flux_mom, x, grad_outputs=torch.ones_like(flux_mom), create_graph=True, retain_graph=True)[0]
        r2 = hu_t + flux_mom_x  # residual for momentum conservation
        return r1, r2


    def compute_bc_loss(self, bc_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Zero-gradient boundary-condition loss: dh/dx = 0 and du/dx = 0 at x=0,Lx.

        The model state is (h, u), so the boundary loss is imposed on those same
        primitive variables. Derivatives are scaled by the training standard
        deviations, matching the data-loss normalization. `lambda_q` is stored as
        `self.norm["lambda_u"]` and is retained as the weight on the second state
        component for compatibility with the experiment interface.
        """
        x = bc_data["x"]
        t = bc_data["t"]
        param = bc_data["param"]

        X_input_bc = torch.cat([x, t, param], dim=1)
        pred_bc = self.model(X_input_bc)
        h_pred, u_pred = self._split_outputs(pred_bc)

        h_x = autograd.grad(
            h_pred,
            x,
            grad_outputs=torch.ones_like(h_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_x = autograd.grad(
            u_pred,
            x,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        e_hx = h_x / self.norm["std_h"]
        e_ux = u_x / self.norm["std_u"]
        lam_u = float(self.norm.get("lambda_u", 1.0))

        return torch.mean(e_hx**2) + lam_u * torch.mean(e_ux**2)



    
    def loss_function(
            self, 
            data: dict[str, torch.Tensor],
            colloc: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss = data + PDE + BC (+ optional penalties)."""
        X_input = torch.cat([data["x"], data["t"], data["param"]], dim=1)

        # ----- Data loss (standardized) -----
        pred = self.model(X_input)
        h_pred, u_pred = self._split_outputs(pred)
        
        # ----- Non-negativity penalty for depth (optional) -----
        # Penalizes negative predicted h on supervised points. This helps prevent unphysical negative depths.
        if getattr(self, "nonneg_h_weight", 0.0) > 0:
            loss_nonneg_h = torch.mean(F.relu(-h_pred))
        else:
            loss_nonneg_h = torch.tensor(0.0, device=self.device)

        # standardize residuals; lambda_q is retained as the public name, but here it weights u relative to h
        e_h = (h_pred - data["h"]) / self.norm["std_h"]
        e_u = (u_pred - data["u"]) / self.norm["std_u"]
        lam_u = float(self.norm.get("lambda_u", 1.0))

        shock_cfg = (self.get_data_params.get('shock_aware_colloc', {}) if hasattr(self, 'get_data_params') else {})
        if bool(shock_cfg.get('balance_dry_loss', False)) and ("is_dry" in data):
            w_dry = float(shock_cfg.get('dry_loss_weight', 3.0))
            w = 1.0 + (w_dry - 1.0) * data["is_dry"]
        else:
            w = 1.0

        if self.l2['data']:
            loss_data = torch.mean(w * (e_h**2)) + lam_u * torch.mean(w * (e_u**2))
        else:
            loss_data = torch.mean(w * torch.abs(e_h)) + lam_u * torch.mean(w * torch.abs(e_u))

        # ----- PDE residual loss at collocation points -----
        if self.pde_weight > 0:
            shock_cfg = (self.get_data_params.get('shock_aware_colloc', {})
                        if hasattr(self, 'get_data_params') else {})
            r1, r2 = self.pde_residual(colloc["x"], colloc["t"], colloc["param"])

            # Optional soft gating + residual scaling (both driven by shock_aware_colloc cfg)
            shock_cfg = (self.get_data_params.get('shock_aware_colloc', {})
                        if hasattr(self, 'get_data_params') else {})

            # Component-wise residual scaling (prevents one PDE from dominating)
            r1n, r2n, _scale_stats = self._normalize_pde_residuals(r1, r2, cfg=shock_cfg)

            # Soft gating: enforce PDE mainly in smooth, wet regions.
            w_gate = self._pde_gating_weights(colloc["x"], colloc["t"], colloc["param"], cfg=shock_cfg)

            if self.l2['pde']:
                loss_pde = torch.mean(w_gate * (r1n**2)) + torch.mean(w_gate * (r2n**2))
            else:
                loss_pde = torch.mean(w_gate * self._rho(r1n)) + torch.mean(w_gate * self._rho(r2n))
        else:
            loss_pde = torch.tensor(0.0, device=self.device)

        # Boundary condition loss: zero-gradient Neumann/outflow condition on h and u.
        loss_bc = self.compute_bc_loss(data["bc"]) if self.bc_weight > 0 else torch.tensor(0.0, device=self.device)

        # Analytic initial-condition loss at t=0.
        loss_ic = self.compute_ic_loss(data["ic"]) if self.ic_weight > 0 else torch.tensor(0.0, device=self.device)

        total_loss = (
            self.data_weight*loss_data
            + self.pde_weight*loss_pde
            + self.bc_weight*loss_bc
            + self.ic_weight*loss_ic
            + self.nonneg_h_weight*loss_nonneg_h
        )
        return total_loss, loss_data, loss_pde, loss_bc, loss_ic


    def save_checkpoint(self, epoch: int, train: bool = False) -> None:
        """Save a model+optimizer checkpoint (train-best or final)."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': getattr(self, 'train_losses', None),
            'val_losses': getattr(self, 'val_losses', None)
        }
        if train:
            torch.save(checkpoint, self.checkpoint_folder + "/checkpoints/train_checkpoint.pth")
        else:
            torch.save(checkpoint, self.checkpoint_folder + "/checkpoints/final_checkpoint.pth")

    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint and restore model+optimizer states."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1  # resume from next epoch
        self.train_losses = checkpoint.get('train_losses', {'general':[], 'data':[], 'pde':[], 'bc':[], 'ic':[]})
        self.val_losses = checkpoint.get('val_losses', {'general':[], 'data':[], 'pde':[], 'bc':[], 'ic':[]})
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.start_epoch}")
        print(f"Be careful, there is no optimizer or model check before loading yet.")

    
    def train(self) -> None:
        """Train the PINN model for the configured number of epochs."""
        os.makedirs(self.checkpoint_folder+'/checkpoints', exist_ok=True)
        if not hasattr(self, 'start_epoch'):
            self.start_epoch = 0

        num_plots = 2
        titles = ['General Loss', 'Data Loss']
        loss_keys = ['general', 'data']
        if self.pde_weight>0:
            num_plots += 1
            titles.append('PDE Loss')
            loss_keys.append('pde')
        if self.bc_weight>0:
            num_plots += 1
            titles.append('Boundary Condition Loss')
            loss_keys.append('bc')
        if self.ic_weight>0:
            num_plots += 1
            titles.append('Initial Condition Loss')
            loss_keys.append('ic')
        
        fig, axs = plt.subplots(1, num_plots, figsize=(4*(num_plots+1), 6))
        fig.suptitle(f'LOG:')
        for ax, title in zip(axs[:num_plots], titles):
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
            ax.grid(True)
        train_lines = [axs[i].plot([], [], label='Training', color='black')[0] for i in range(num_plots)]
        val_lines = [axs[i].plot([], [], label='Validation', color='red', alpha=0.8)[0] for i in range(num_plots)]
        for ax in axs[:num_plots]:
            ax.legend()

        plt.tight_layout()
        if self.display:
            from IPython.display import display
            display_handle = display(fig, display_id=True)

        if not hasattr(self, 'train_losses'):
            self.train_losses = {'general': [], 'data': []}
            if self.pde_weight>0:
                self.train_losses['pde'] = []
            if self.bc_weight>0:
                self.train_losses['bc'] = []
            if self.ic_weight>0:
                self.train_losses['ic'] = []
        if not hasattr(self, 'val_losses'):
            self.val_losses = {'general': [], 'data': []}
            if self.pde_weight>0:
                self.val_losses['pde'] = []
            if self.bc_weight>0:
                self.val_losses['bc'] = []
            if self.ic_weight>0:
                self.val_losses['ic'] = []
        best_val_loss = float('inf')
        if self.rampup_epochs_share:
            rampup_epochs = self.rampup_epochs_share * self.num_epochs
            target_pde_weight = self.pde_weight
            target_bc_weight = self.bc_weight
            target_ic_weight = self.ic_weight

        start_time = time.perf_counter()
        train_step_seconds = 0.0
        validation_seconds = 0.0

        self.model.train()
        for epoch in range(self.start_epoch, self.num_epochs+self.start_epoch):
            if self.rampup_epochs_share and epoch < rampup_epochs:
                frac = epoch / rampup_epochs
                self.pde_weight = 0.1*target_pde_weight + (target_pde_weight - 0.1*target_pde_weight)*frac
                self.bc_weight = 0.1*target_bc_weight + (target_bc_weight - 0.1*target_bc_weight)*frac
                self.ic_weight = 0.1*target_ic_weight + (target_ic_weight - 0.1*target_ic_weight)*frac

            _train_step_start = time.perf_counter()
            self.optimizer.zero_grad()
            train_loss, train_loss_data, train_loss_pde, train_loss_bc, train_loss_ic = self.loss_function(
                data=self.data["train"],
                colloc=self.colloc["train"]
            )
            train_loss.backward()

            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()**2
            total_norm = total_norm**0.5

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            train_step_seconds += time.perf_counter() - _train_step_start

            self.train_losses['general'].append(train_loss.item())
            self.train_losses['data'].append(train_loss_data.item())
            if self.pde_weight>0:
                self.train_losses['pde'].append(train_loss_pde.item())
            if self.bc_weight>0:
                self.train_losses['bc'].append(train_loss_bc.item())
            if self.ic_weight>0:
                self.train_losses['ic'].append(train_loss_ic.item())

            # validation
            _validation_start = time.perf_counter()
            self.model.eval()
            val_loss, val_loss_data, val_loss_pde, val_loss_bc, val_loss_ic = self.loss_function(
                data=self.data["val"],
                colloc=self.colloc["val"]
            )
            self.model.train()
            validation_seconds += time.perf_counter() - _validation_start
            self.val_losses['general'].append(val_loss.item())
            self.val_losses['data'].append(val_loss_data.item())
            if self.pde_weight>0:
                self.val_losses['pde'].append(val_loss_pde.item())
            if self.bc_weight>0:
                self.val_losses['bc'].append(val_loss_bc.item())
            if self.ic_weight>0:
                self.val_losses['ic'].append(val_loss_ic.item())

            log_every = max(1, int(self.num_epochs * 0.01))
            if (epoch % log_every) == 0 or epoch == (self.num_epochs + self.start_epoch - 1):
                # update loss plots
                fig.suptitle(
                f"LOG: grad_norm_gross={total_norm:.3f}, Data_w={self.data_weight:.3f}, PDE_w={self.pde_weight:.3f}, "
                f"BC_w={self.bc_weight:.3f}, IC_w={self.ic_weight:.3f}, "
                f"{self.optimizer.__class__.__name__}_lr={self.optimizer.param_groups[0]['lr']:.8f}"
                )
                for i, k in enumerate(loss_keys):
                    _train = self.train_losses[k]
                    _val = self.val_losses[k]
                    train_lines[i].set_data(np.arange(len(_train)), _train)
                    val_lines[i].set_data(np.arange(len(_val)), _val)
                for ax in axs[:num_plots]:
                    ax.relim(); ax.autoscale_view(True, True, True)

                if self.display:
                    display_handle.update(fig)

                # checkpoint as before
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    self.save_checkpoint(epoch, train=True)
                    try:
                        fig.savefig(f'./{self.checkpoint_folder}/train_checkpoint.png')
                    except RecursionError as e:
                        print(f"[warn] savefig recursion error (train_checkpoint) -> skipping: {e}")
            
        self.start_epoch = self.num_epochs
        self.save_checkpoint(self.num_epochs, train=False)
        elapsed_time = time.perf_counter() - start_time
        self.timings.add('training_total', elapsed_time)
        self.timings.add('training_gradient_steps', train_step_seconds)
        self.timings.add('validation_total', validation_seconds)
        self.timings.save(os.path.join(self.checkpoint_folder, 'timing_summary.json'), extra={
            'num_epochs': self.num_epochs,
            'seconds_per_epoch': elapsed_time / max(self.num_epochs, 1),
        })
        print(f"Training and validation completed in {elapsed_time:.2f} seconds "
              f"({elapsed_time/max(self.num_epochs,1):.3f} s/epoch).")
        try:
            fig.savefig(f'./{self.checkpoint_folder}/train_{self.num_epochs}e_Tfinal={self.t_final}_totalTime={elapsed_time}s.png')
        except RecursionError as e:
            print(f"[warn] savefig recursion error (final train plot) -> skipping: {e}")
        plt.close('all')



    def _predict_fields_on_grid(self, x: np.ndarray, t: float, h_left: float, h_right: float) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        N = x.shape[0]

        x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=self.device)
        t_t = torch.full((N, 1), fill_value=float(t), dtype=torch.float32, device=self.device)

        p_np = np.tile([float(h_left), float(h_right)], (N, 1))
        p_np = self._to_invariants(p_np)
        p_t = torch.tensor(p_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            inp = torch.cat([x_t, t_t, p_t], dim=1)
            pred = self.model(inp)
            h_pred_t, u_pred_t = self._split_outputs(pred)
            h_pred = h_pred_t[:, 0].detach().cpu().numpy()
            u_pred = u_pred_t[:, 0].detach().cpu().numpy()
            q_pred = self._q_from_hu_np(h_pred, u_pred)

        return {
            "x": x,
            "h_pred": h_pred,
            "u_pred": u_pred,
            "q_pred": q_pred,
        }


    def evaluate(
        self,
        test_t: float = 1.0,
        test_h: tuple[float, float, str] = (12.0, 5.0, 'perpendicular'),
        plot: bool = True
    ) -> dict[str, Any]:
        """Evaluate the trained PINN on one regime/time and save diagnostic plots."""
        return plotting_utils.evaluate_model(self, test_t=test_t, test_h=test_h, plot=plot)
    

    @staticmethod
    def _fdx_numpy(y: np.ndarray, dx: float) -> np.ndarray:
        """1D finite difference derivative with one-sided ends."""
        dy = np.empty_like(y)
        dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
        dy[0] = (y[1] - y[0]) / dx
        dy[-1] = (y[-1] - y[-2]) / dx
        return dy


    def compute_space_time_rel_errors(
        self,
        h_left: float,
        h_right: float,
        dam_shape: str = 'perpendicular',
        Lx: float = 100.0,
        Nx: int = 402,
        t_final: float = 2.5,
        dt: float = 1e-4,
        sample_interval: float = 0.01,
        include_H1: bool = True,
        field: str = "h",  # "h", "q", or "state"
    ) -> dict[str, float]:
        """Compute tROM-style space-time relative errors.

        The tROM paper defines the error for the full state u in the space-time norms
        L^2(0,T;L^2(\\Omega)) and L^2(0,T;H^1(\\Omega)).
        Here we support three options:
          - field='h'     : scalar depth error
          - field='q'     : scalar discharge error
          - field='state' : vector-valued state error using components (h, q)

        For field='state', the discrete norms are computed by summing component energies,
        i.e. ||(h,q)||^2 := ||h||^2 + ||q||^2, and similarly for H^1.
        """
        field = str(field).lower()
        if field not in {"h", "q", "state"}:
            raise ValueError(f"field must be one of {{'h','q','state'}}, got: {field}")

        self.model.eval()
        snaps = simulate_shallow_water(
            h_left=h_left, h_right=h_right, dam_shape=dam_shape,
            Lx=Lx, Nx=Nx, t_final=t_final, dt=dt,
            sample_interval=sample_interval, plot=False
        )
        if len(snaps) < 2:
            raise RuntimeError(
                f"Need at least 2 snapshots to integrate in time; got {len(snaps)}. "
                f"Check t_final={t_final} and sample_interval={sample_interval}."
            )

        dx = Lx / (Nx - 2)
        num_L2 = den_L2 = 0.0
        num_H1 = den_H1 = 0.0

        prev_t = snaps[0]['t']
        for snap in snaps[1:]:
            t = float(snap['t'])
            dtw = t - prev_t
            prev_t = t
            x = np.asarray(snap['x']).reshape(-1)

            h_true = np.asarray(snap['h']).reshape(-1)
            u_true = np.asarray(snap['u']).reshape(-1)
            q_true = h_true * u_true

            pred_fields = self._predict_fields_on_grid(
                x=x,
                t=t,
                h_left=h_left,
                h_right=h_right,
            )
            h_pred = pred_fields["h_pred"]
            q_pred = pred_fields["q_pred"]

            if field == "h":
                y_true_list = [h_true]
                y_pred_list = [h_pred]
            elif field == "q":
                y_true_list = [q_true]
                y_pred_list = [q_pred]
            else:  # state = (h, q)
                y_true_list = [h_true, q_true]
                y_pred_list = [h_pred, q_pred]

            step_num_L2 = 0.0
            step_den_L2 = 0.0
            step_num_H1 = 0.0
            step_den_H1 = 0.0
            for y_true, y_pred in zip(y_true_list, y_pred_list):
                e = y_pred - y_true
                step_num_L2 += float(e @ e)
                step_den_L2 += float(y_true @ y_true)

                if include_H1:
                    e_x = self._fdx_numpy(e, dx)
                    y_x = self._fdx_numpy(y_true, dx)
                    step_num_H1 += float((e @ e) + (e_x @ e_x))
                    step_den_H1 += float((y_true @ y_true) + (y_x @ y_x))

            num_L2 += step_num_L2 * dx * dtw
            den_L2 += step_den_L2 * dx * dtw
            if include_H1:
                num_H1 += step_num_H1 * dx * dtw
                den_H1 += step_den_H1 * dx * dtw

        out = {"E_L2_space_time": float(np.sqrt(num_L2) / (np.sqrt(den_L2) + 1e-30))}
        if include_H1:
            out["E_H1_space_time"] = float(np.sqrt(num_H1) / (np.sqrt(den_H1) + 1e-30))
        return out

    def _relative_error_time_series(
        self,
        h_left: float,
        h_right: float,
        dam_shape: str = 'perpendicular',
        Lx: float = 100.0,
        Nx: int = 402,
        t_final: float = 2.5,
        dt: float = 1e-4,
        sample_interval: float = 0.01,
        field: str = "h",
    ) -> dict[str, np.ndarray]:
        """Return time series of relative L2-in-space errors for h, q, or the full state (h,q)."""
        field = str(field).lower()
        if field not in {"h", "q", "state"}:
            raise ValueError(f"field must be one of {{'h','q','state'}}, got: {field}")

        self.model.eval()
        snaps = simulate_shallow_water(
            h_left=h_left, h_right=h_right, dam_shape=dam_shape,
            Lx=Lx, Nx=Nx, t_final=t_final, dt=dt,
            sample_interval=sample_interval, plot=False
        )

        times = []
        rel_errors = []
        final_payload = None

        for snap in snaps:
            t = float(snap['t'])
            x = np.asarray(snap['x']).reshape(-1)
            h_true = np.asarray(snap['h']).reshape(-1)
            u_true = np.asarray(snap['u']).reshape(-1)
            q_true = h_true * u_true

            pred_fields = self._predict_fields_on_grid(
                x=x,
                t=t,
                h_left=h_left,
                h_right=h_right,
            )
            h_pred = pred_fields["h_pred"]
            q_pred = pred_fields["q_pred"]

            if field == "h":
                y_true_list = [h_true]
                y_pred_list = [h_pred]
            elif field == "q":
                y_true_list = [q_true]
                y_pred_list = [q_pred]
            else:
                y_true_list = [h_true, q_true]
                y_pred_list = [h_pred, q_pred]

            num = 0.0
            den = 0.0
            pred_energy = 0.0
            for y_true, y_pred in zip(y_true_list, y_pred_list):
                e = y_pred - y_true
                num += float(e @ e)
                den += float(y_true @ y_true)
                pred_energy += float(y_pred @ y_pred)

            abs_err = float(np.sqrt(num))
            ref_norm = float(np.sqrt(den))
            pred_norm = float(np.sqrt(pred_energy))

            if ref_norm <= 1e-12:
                rel = np.nan
            else:
                rel = float(abs_err / ref_norm)

            times.append(t)
            rel_errors.append(rel)
            final_payload = {
                "x": x,
                "h_true": h_true,
                "q_true": q_true,
                "h_pred": h_pred,
                "q_pred": q_pred,
                "t": t,
            }

        return {
            "t": np.asarray(times, dtype=float),
            "rel_error": np.asarray(rel_errors, dtype=float),
            "final": final_payload,
        }


    def paper_style_eval(
        self,
        sampling: str = "mc", # "mc" or "grid"
        n_samples: int = 99, # used if sampling == "mc"
        seed: int = 123,
        include_H1: bool = False,
        Lx: float = 100.0,
        Nx: int = 402,
        t_final: float = 2.5,
        dt: float = 1e-4,
        sample_interval: float = 0.01,
        save_name: str = "paper_eval",
        grid_hL: list[float] | None = None,
        grid_hR: list[float] | None = None,
        dam_shape: str = 'perpendicular',
        aggregate: str = "mean",  # "mean" (paper-style) or "integral"
        fields: list[str] | None = None,  # e.g., ["h"], ["h","q"]
    ) -> dict:
        """
        Compute paper-style metrics over a parameter set and save CSV/JSON.
        - If sampling == "grid", use the paper's h_L & h_R lists or grid_hL/grid_hR if they are provided.
        - If sampling == "mc", sample uniformly in [10,28] x [0,8].
        """
        if fields is None:
            fields = ["h"]
        fields = [str(f).lower() for f in fields]
        for f in fields:
            if f not in {"h", "q", "state"}:
                raise ValueError(f"Invalid field in fields: {f}. Must be one of 'h','q','state'.")
        aggregate = str(aggregate).lower()
        if aggregate not in {"mean", "integral", "both"}:
            raise ValueError(f"aggregate must be 'mean', 'integral', or 'both', got: {aggregate}")

        if grid_hL is None:
            grid_hL = np.linspace(10.0, 28.0, 20)
        if grid_hR is None:
            grid_hR = np.linspace(0.0, 8.0, 25)
        params = [(float(hL), float(hR)) for hL in grid_hL for hR in grid_hR]

        if sampling == "mc":
            rng = np.random.default_rng(seed)
            if n_samples <= len(params):
                idx = rng.choice(len(params), size=n_samples, replace=False)
            else:
                idx = rng.integers(0, len(params), size=n_samples)
            params = [params[i] for i in idx]

        rows = []
        # per-field accumulators
        E_vals = {f: [] for f in fields}
        E_H1_vals = {f: [] for f in fields} if include_H1 else None

        for (hL, hR) in params:
            row = {"h_left": hL, "h_right": hR}
            for f in fields:
                met = self.compute_space_time_rel_errors(
                    h_left=hL, h_right=hR, dam_shape=dam_shape, Lx=Lx, Nx=Nx,
                    t_final=t_final, dt=dt, sample_interval=sample_interval,
                    include_H1=include_H1, field=f
                )
                row[f"{f}_E_L2_space_time"] = met["E_L2_space_time"]
                E_vals[f].append(met["E_L2_space_time"])
                if include_H1:
                    row[f"{f}_E_H1_space_time"] = met["E_H1_space_time"]
                    E_H1_vals[f].append(met["E_H1_space_time"])
            rows.append(row)

        # Parameter-domain measure for converting mean -> integral.
        # For a full 2D grid, this is (ΔhL * ΔhR). For a slice where one dimension is fixed
        # (e.g., len(grid_hL)==1), we use the non-degenerate dimension only so the measure
        # does not collapse to 0.
        grid_hL_arr = np.asarray(grid_hL, dtype=float)
        grid_hR_arr = np.asarray(grid_hR, dtype=float)
        d_hL = float(np.max(grid_hL_arr) - np.min(grid_hL_arr)) if grid_hL_arr.size > 0 else 0.0
        d_hR = float(np.max(grid_hR_arr) - np.min(grid_hR_arr)) if grid_hR_arr.size > 0 else 0.0
        ranges = []
        if d_hL > 1e-12:
            ranges.append(d_hL)
        if d_hR > 1e-12:
            ranges.append(d_hR)
        area = float(np.prod(ranges)) if len(ranges) > 0 else 0.0

        summary = {
            "sampling": sampling,
            "num_params": len(params),
            "aggregate": aggregate,
            "fields": fields,
        }

        for f in fields:
            mean_L2 = float(np.mean(E_vals[f]))
            sup_L2  = float(np.max(E_vals[f]))
            summary[f"{f}_E_L2(L2)_avg_mean"] = mean_L2
            summary[f"{f}_E_L2(L2)_avg_integral"] = float(area * mean_L2)
            summary[f"{f}_E_L2(L2)_sup"] = sup_L2
            summary[f"{f}_E_L2(L2)_mean_unscaled"] = mean_L2

            if include_H1:
                mean_H1 = float(np.mean(E_H1_vals[f]))
                sup_H1  = float(np.max(E_H1_vals[f]))
                summary[f"{f}_E_L2(H1)_avg_mean"] = mean_H1
                summary[f"{f}_E_L2(H1)_avg_integral"] = float(area * mean_H1)
                summary[f"{f}_E_L2(H1)_sup"] = sup_H1
                summary[f"{f}_E_L2(H1)_mean_unscaled"] = mean_H1

        # Save
        import csv, json, pathlib
        out_dir = pathlib.Path(self.checkpoint_folder) / "paper_style_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{save_name}.csv"
        json_path = out_dir / f"{save_name}_summary.json"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        with open(json_path, "w") as f:
            summary["aggregate"] = aggregate
            json.dump(summary, f, indent=2)

        print(f"[paper_eval] Saved per-parameter metrics to {csv_path}")
        print(f"[paper_eval] Summary: {summary}")
        return {"rows": rows, "summary": summary}


    def plot_trom_style_final_profiles(
        self,
        test_h: list[tuple[float, float, str]],
        test_t: float | list[float] = 2.5,
        field: str = "h",
        plot_kind: str = "profile",
        Lx: float | None = None,
        Nx: int | None = None,
        dt: float | None = None,
        save_name: str | None = None,
    ) -> list[str]:
        """Create tROM-style multi-panel figures for final-time profiles or spectra."""
        return plotting_utils.plot_trom_style_final_profiles(
            self,
            test_h=test_h,
            test_t=test_t,
            field=field,
            plot_kind=plot_kind,
            Lx=Lx,
            Nx=Nx,
            dt=dt,
            save_name=save_name,
        )


    def plot_trom_style_error_evolution(
        self,
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
        """Create a multi-panel relative-error-vs-time figure."""
        return plotting_utils.plot_trom_style_error_evolution(
            self,
            test_h=test_h,
            field=field,
            Lx=Lx,
            Nx=Nx,
            t_final=t_final,
            dt=dt,
            sample_interval=sample_interval,
            save_name=save_name,
            extra_table_h=extra_table_h,
        )


    def test_model(
        self,
        test_t: list[float] = [0.5, 1.0, 1.5, 2.0],
        test_h: list[tuple[float, float, str]] = [
            (3.0, 0.0, 'perpendicular'),
            (5.0, 0.0, 'perpendicular'),
            (7.0, 0.0, 'perpendicular'),
            (10.0, 0.0, 'perpendicular'),
            (12.0, 0.0, 'perpendicular'),
            (15.0, 0.0, 'perpendicular'),
            (17.0, 0.0, 'perpendicular'),
        ],
        plot: bool = False,
        paper_eval_kwargs: dict | None = None
    ) -> None:
        """Run evaluation across multiple times/regimes and save a JSON summary."""
        return plotting_utils.test_model(
            self,
            test_t=test_t,
            test_h=test_h,
            plot=plot,
            paper_eval_kwargs=paper_eval_kwargs,
        )