---
name: Sub-Experiment
about: Child issue for one specific experimental run within an experiment group
title: "[Sub-Experiment] "
labels: extension
assignees: ''
---

**Parent:** #

## Key finding
<!-- One sentence. Written after evaluation completes — this is what goes straight into an email. -->
> _Pending evaluation_

## Hypothesis
<!-- What do we expect this specific variant to show, and why -->

## Setup
| Parameter | This variant | Baseline (32-dim / Phase G) |
|---|---|---|
| Bottleneck dim |  | 32 |
| Temporal stride |  | 20 |
| Curriculum |  | A → B → C → D → E → F → G |
| Everything else |  | unchanged |

## Execution checklist
- [ ] Modify config
- [ ] Train through curriculum
- [ ] Checkpoint saved
- [ ] Evaluate (R-D sweep + entropy analysis)
- [ ] Analyze / write up results below

## Results
| Metric | This variant | 32-dim baseline | Delta |
|---|---|---|---|
| Bitrate (kbps) |  | 5.90 |  |
| PESQ-WB |  | 1.256 |  |
| STOI |  | 0.756 |  |
| Latent entropy (bits) |  | ~1.520 |  |

## Artifacts
| Output | Path |
|---|---|
| Checkpoint |  |
| Plots |  |

## Interpretation
<!-- What this means for the parent question -->

## Open questions / follow-up
