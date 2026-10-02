# Parametric SWE PINN

Source code accompanying:

**Comparison of a Parametric Physics-Informed Neural Network and a Tensorial Reduced-Order Model for the Shallow-Water Dam-Break Problem**

Anton Myshak, Md Rezwan Bin Mizan, Ilya Timofeyev

Accepted for publication in *Fluids* (MDPI).

[arXiv:2607.27433](https://arxiv.org/abs/2607.27433)

This repository contains the implementation, training configuration, and
evaluation pipeline used for the numerical experiments reported in the paper.

## Repository structure

- `src/run_experiments.py` — single CLI entry point; legacy CLI remains supported.
- `src/SWE_PINN_public.py` — PINN, physics residuals/losses, data preparation, training, and metric kernels.
- `src/data_generator.py` — Local Lax–Friedrichs/Rusanov FOM used by the submitted work.
- `src/architectures.py` — only the architectures actually supported by the public runner (`lin`, `residual_lin`).
- `src/evaluation_pipeline.py` — generic JSON-driven evaluation orchestration.
- `src/plotting_utils.py` — plotting/legacy evaluation helpers; error-evolution layouts now accept arbitrary case counts.
- `src/experiment_utils.py` — seeding, timing, source snapshots, resolved configs, system metadata.
- `configs/paper_training_grid.json` — submitted 65-regime training grid.
- `configs/paper_training_args.json` — submitted model setup.
- `configs/paper_evaluation.json` — example generic evaluation plan reproducing the main paper-style evaluation products.

## Experiment outputs

Each run creates an `experiments/<timestamp>/` directory containing:

- `resolved_config.json` — resolved CLI/configuration;
- `command.txt` — launch command;
- `system_info.json` — Python/PyTorch/NumPy/platform/device metadata;
- `scripts/` — snapshot of the Python and JSON files used for the run;
- `checkpoints/` — best-validation and final checkpoints;
- `timing_summary.json` — FOM generation, shock preprocessing, collocation, training-gradient, validation and total-training timings;
- `pipeline_timing.json` for normal full-pipeline runs;
- `test_results.json` — relative-error results for the standard evaluation cases and times;
- `eval_mc_paper.csv` and `eval_mc_paper.json` — paper-style Monte Carlo evaluation metrics when paper evaluation is enabled;
- parameter-slice evaluation CSV/JSON summaries when slice evaluation is enabled;
- training and validation loss-history figures;
- shock-aware collocation sanity/diagnostic figures;
- solution-profile and Fourier-spectrum comparison figures for requested test cases;
- relative-error evolution figures for water depth `h` and discharge `q`;
- paper-specific profile, spectrum, and parameter-slice figures requested by the evaluation configuration.

## Quick reproducibility test

The following command performs one full-batch gradient step over the submitted data/collocation sets, and exercises shock-aware sampling, PDE gating, training, checkpointing, timing and plots:

```bash
python src/run_experiments.py \
  --output_dir ../ \
  --seed 42 \
  --train_grid_json ./configs/final_model.json \
  --val_regime 17.0,4.0,perpendicular \
  --val_sample_interval 0.25 \
  --model lin \
  --layers 4 120 120 120 120 120 2 \
  --activation SiLU \
  --num_epochs 1 \
  --disable_scheduler \
  --optimizer_lr 1e-3 \
  --data_weight 1.0 \
  --pde_weight 0.3 --l2_pde \
  --bc_weight 0.6 --ic_weight 1.0 \
  --enforce_pos_h none --nonneg_h_weight 0.01 \
  --lambda_q 5.0 \
  --rampup_epochs_share 0.25 \
  --N_colloc_train 80000 --N_colloc_val 20000 \
  --shock_aware_colloc --shock_indicator h \
  --shock_w_excl 2.5 --shock_w_focus 0.0 --shock_focus_share 0.0 \
  --dry_hR_eps 0.5 --shock_smooth_window 7 --shock_boundary_buf_frac 0.02 \
  --pde_gate --gate_alpha_grad 8.0 --gate_h_min 0.05 --gate_floor 0.05 \
  --residual_scaling ema_rms --residual_ema_beta 0.99 --residual_scale_eps 1e-8 \
  --dry_loss_weight 1.0 \
  --eval_config_json ./configs/paper_evaluation.json
```

On Apple Silicon you may separately test MPS with `--device mps`. However, CPU works faster for some reason.

## Reproducing the paper configuration

The submitted model was trained for 20,000 epochs using the configuration
provided in `configs/paper_training_args.json`.

Run:

```bash
python src/run_experiments.py \
  --config configs/paper_training_args.json \
  --eval_config_json configs/paper_evaluation.json
```

## Generic evaluation plan

`--eval_config_json` is the preferred path for new work. A plan can specify:

- arbitrary parameter cases and times;
- paper-style aggregate metrics;
- profile/spectrum plots;
- error-evolution plots.

## Representative experiment

The `experiments/` directory contains the outputs of a representative training
run with seed 42 as an example. The remaining seed runs used for the reported multi-seed statistics are not
included because they largely duplicate generated checkpoints and evaluation
artifacts. They can be reproduced by running the provided configuration with
the corresponding seeds.

## Citation

If you use this code, please cite the accompanying preprint:

```bibtex
@article{myshak2026parametric,
  title={Comparison of a Parametric Physics-Informed Neural Network and a Tensorial Reduced-Order Model for the Shallow-Water Dam-Break Problem},
  author={Myshak, Anton and Mizan, Md Rezwan Bin and Timofeyev, Ilya},
  journal={arXiv preprint arXiv:2607.27433},
  year={2026},
  doi={10.48550/arXiv.2607.27433}
}