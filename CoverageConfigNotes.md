# Coverage / Routing Metric Notes

This note traces the routing metrics printed during training back to their exact `model.py` stat sources and the `config.py` knobs that most directly affect them.

I also tightened the nearby log labels in `train.py` so the printed names now reflect what the values actually are:

- `cov` -> `torus_cov`
- `breadth` -> `cell_breadth`
- `soft_raw` -> `soft_active`
- `soft_breadth` -> `soft_cell_breadth`
- `conc` -> `usage_conc`
- `eff` -> `eff_count`
- `ccv` -> `cell_cov`

The validation print uses the same clearer naming (`val_active`, `val_soft_active`, `val_eff_count`, `val_cell_breadth`, `val_soft_cell_breadth`, `val_cell_cov`, `val_usage_entropy`, `val_usage_conc`).

## Metric Map

| Printed label | What it measures | Exact `model.py` source chain | Most direct `config.py` knobs | Notes |
| --- | --- | --- | --- | --- |
| `torus_cov` | Torus coverage loss | `PrismalTorusCore.step()` computes `active_balance_loss = abs(soft_active_fraction - active_target_fraction) / active_target_fraction`; `PrismalWaveModel._forward_torus_path()` aggregates it as `cell_coverage_loss`; `PrismalWaveModel._forward_torus()` exposes it as `torus_coverage_loss` and `emitter_cell_coverage_loss` | `torus_activity_threshold`, `torus_active_target_fraction`, `torus_active_balance_weight` | `torus_cov` and `cell_cov` are the same scalar under two aliases. |
| `cell_breadth` | Hard active-cell fraction | `PrismalTorusCore.step()` counts `active_cells = (cell_energy > activity_threshold).sum(...)`; `_forward_torus_path()` normalizes to `emitter_cell_breadth`; `_forward_torus()` passes it through | `torus_activity_threshold`, `torus_depth`, `torus_height`, `torus_width` | This is the hard occupancy fraction, not a count. |
| `soft_active` | Soft active-cell count | `PrismalTorusCore.step()` computes `soft_active_cells = sigmoid((cell_energy - activity_threshold) * 12).sum(...)`; `_forward_torus_path()` aggregates `emitter_cell_soft_occupancy`; `_forward_torus()` passes it through | `torus_activity_threshold`, `torus_depth`, `torus_height`, `torus_width` | This is the unnormalized soft count, so the old `soft_raw` label was misleading. |
| `soft_cell_breadth` | Soft active-cell fraction | Same path as above, but normalized in `PrismalTorusCore.step()` as `soft_active_fraction = soft_active_cells / total_cells` and exposed as `emitter_cell_soft_breadth` | `torus_activity_threshold`, `torus_depth`, `torus_height`, `torus_width` | This is the normalized counterpart to `soft_active`. |
| `usage_conc` | Concentration of stencil usage | `PrismalTorusCore.step()` computes `stencil_usage_concentration = stencil_weights.square().sum(dim=-1).mean() * stencil_weights.size(-1)`; `_forward_torus_path()` averages `emitter_usage_concentration`; `_forward_torus()` passes it through | `torus_local_field_radius`, `torus_relay_write_radius`, `torus_inner_temperature`, `torus_outer_temperature` | The support size comes from the local/relay write radii; the coordinate temperature helper shapes the sharpness of the stencil. |
| `eff_count` | Entropy-derived effective stencil count | `PrismalTorusCore.step()` computes `stencil_effective_count = _effective_count_from_weights(stencil_weights).mean()`; `_forward_torus_path()` stores it as `emitter_cell_effective_count`; `_forward_torus()` passes it through | `torus_local_field_radius`, `torus_relay_write_radius`, `torus_inner_temperature`, `torus_outer_temperature`, `emitter_mixture_target_count` | `_effective_count_from_weights()` returns `exp(entropy(weights))`, so `eff_count` is not a literal top-k size. `emitter_mixture_target_count` only affects the auxiliary penalty, not the printed value itself. |
| `cell_cov` | Cell coverage loss alias | Same tensor as `torus_cov`, but read through the `emitter_cell_coverage_loss` alias in `PrismalWaveModel._forward_torus_path()` / `_forward_torus()` | Same as `torus_cov` | This label exists only to show the alternate alias; the value is identical. |

## Practical Readout

- If `torus_cov` or `cell_cov` moves, check `torus_activity_threshold` and `torus_active_target_fraction` first.
- If `cell_breadth` or `soft_cell_breadth` shifts, the most likely cause is the activity threshold or torus grid size.
- If `usage_conc` or `eff_count` shifts, look at the torus write radii and the torus temperature curve before anything else.

