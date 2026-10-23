# Replication of *The Price of Failure Risk*

This repository reproduces the numerical findings of the paper *The Price of Failure Risk: Clipped Barrier Policy
Gradients with Convex Regularization* (Malo, Viitasaari, Suominen, Vilkkumaa, Tahvonen). It holds every study's
simulator, training and analysis scripts, the protocols fixed before each campaign, and the records of every run, so
that each table and figure of the paper can be regenerated from the records, and each campaign can be run again.

It is separate from the toolbox. The toolbox, [`barrierpg`](https://github.com/pma-papers/barrierpg), is for applying
clipped barrier policy gradients to a new problem; this repository is for checking the paper. Results are named by the
titles the paper gives them, not by statement, section or table numbers, which change between revisions. The tables
the scripts generate were checked against revision 73 of the paper.

## Studies

| Folder | Study in the paper | What it tests |
|---|---|---|
| [`wear-allowance/`](wear-allowance) | An asset operated under a wear allowance; the prior on a learned controller; the roles of the regularizers on PPO | The price against an exact benchmark, the adaptive multiplier, the interior barrier under a misjudged allowance, the action anchor, the parameter anchor and parameter noise |
| [`fuel-tank/`](fuel-tank) | A learned controller with a fuel tank | The price on a learned controller (LunarLander with a fuel tank), the dual step of an adaptive multiplier |
| [`solvable-flow/`](solvable-flow) | The convergence theorem and the prior on a solvable flow; the roles of the regularizers | The slack of the curvature threshold, the prior policy, noise, pull and spread, the particle form |
| [`benchmark-suite/`](benchmark-suite) | A task of a benchmark suite (details in Online Resource 1) | A Safety-Gymnasium navigation task: the price, the adaptive multiplier and the expected-cost formulation |
| [`lib/`](lib) | shared modules | plotting style, density flow, digest helpers |

Each folder has the protocols (`PROTOCOL*.md`, some with a SHA256 recorded before the campaign), the runners, the
run records (`campaign*/`, `runs/`, `lunar/`, `safety/`, JSON and logs), the trained policies where a re-evaluation
needs them (`.zip`, with the normalization statistics in `.pkl`), and the analysis scripts.

[`docs/paper-studies.md`](docs/paper-studies.md) walks through the studies as a practitioner would run them with the
toolbox: the problem in the paper's notation, the configuration choices and why, the paper's numbers, and what to take
from them.

## Regenerating the tables and figures

Run each script from its own folder; every script finds its inputs relative to that folder. The manuscript file that
each output corresponds to is named in the last column.

| Result | Script | Python | Manuscript file |
|---|---|---|---|
| Wear allowance: exact benchmark and campaign tables, margin figure | `wear-allowance/analyze_v41.py` | `constrained-rl` | `tables/wear-benchmark`, `tables/wear-campaign`, `figures/wear-margin` |
| Wear allowance: frontier, multiplier and anchor figures | `wear-allowance/relabel_wear_figures_v47.py` | `constrained-rl` | `figures/wear-frontier`, `wear-multiplier`, `parameter-anchor` |
| The prior on a learned controller: table | `wear-allowance/anchor_analyze_v41.py` | `constrained-rl` | `tables/parameter-anchor` |
| Roles of the regularizers on PPO: table and figure | `wear-allowance/analyze_noise_v60.py` | `constrained-rl` | `tables/wear-noise`, `figures/wear-noise` |
| Roles of the regularizers on PPO, the pull decoupled from Adam (2026-10-20, after revision 68) | `wear-allowance/analyze_decoupled_v68.py` | `constrained-rl` | no table of its own; `wear-allowance/REPORT-decoupled-v68.md` |
| Roles of the regularizers on PPO under three optimizers, with the amendment's rows (2026-10-20, after revision 68) | `wear-allowance/analyze_optimizers_v68.py` (per-coordinate records: `coordinate_diagnostic_v68.py`, `constrained-rl-numerics`) | `constrained-rl` | no table of its own; `wear-allowance/REPORT-optimizers-v68.md`, `wear-allowance/wear-optimizers-v68.tex` |
| Fuel tank: tables and figure | `fuel-tank/lunar_analyze_v41.py` | `constrained-rl` | `tables/fuel-tank`, `tables/fuel-tank-paired`, `figures/fuel-tank` |
| Solvable flow: convergence test | `solvable-flow/regularizer_figure_v46.py` | `constrained-rl` | `tables/flow-test`, `figures/flow-test` |
| Solvable flow: prior strength | `solvable-flow/prior_figure.py` | `constrained-rl` | `tables/prior-sweep`, `figures/prior-sweep` |
| Solvable flow: regularization paths | `solvable-flow/prior_figure_v47.py`, `prior_path_table_v41.py` | `constrained-rl` | `figures/prior`, `tables/prior-paths` |
| Solvable flow: particle form | `solvable-flow/particle_flow_v58.py`, `particle_table_v58.py` | `constrained-rl` | `figures/particle-flow`, `tables/particle-flow` |
| Solvable flow: roles of the regularizers | `solvable-flow/roles_flow_v59.py` | `constrained-rl` | `figures/roles-flow`, `tables/roles-flow` |
| Solvable flow: the learning-error bound | `solvable-flow/corollary_bound_v60.py` | `constrained-rl` | `figures/corollary-bound` |
| Benchmark suite: table and figure | `benchmark-suite/analyze_safety_v63.py` (writes to `output/`) | `constrained-rl` | `tables/safety`, `figures/safety` |
| Illustrations of the theory (Figures 1 and 3, the curvature budget) | `illustrations/introduction_schematic.py`, `running_example_figures.py`, `concept_figures.py` | `constrained-rl` | `figures/logic`, `identity`, `curvature` |

`collect_figures.py <folder>` copies every figure above into a manuscript figure folder under the manuscript's
names. The figures are drawn at the journal's print width with sans-serif lettering (`lib/plotstyle.py`). The
generated tables reproduce the manuscript's table files of revision 73 byte for byte, except the table of the roles
of the regularizers on the solvable flow, which was assembled by hand from `solvable-flow/roles-flow-v59.json`. Three
scripts carry the corrections the paper made in revision 70: the reference spread of injected noise under the pull is
`sqrt(kappa_1/kappa_2)` (`analyze_noise_v60.py`; the protocol's design values stay in its check (R1)), the
benchmark suite's validation shares the error `0.05` among all 72 reported candidates (`analyze_safety_v63.py`), and
the solvable flow's curvature constant is called a finite-difference estimate (`regularizer_figure_v41.py`).
The wear-allowance table carries the column "bound": the price–risk law with the idle policy as comparison and
the measured optimization error, computed per seed in `wear-allowance/analyze_v41.py`.

## Running the campaigns again

The launch scripts (`*campaign*.sh`, `*run_one*.sh`, `launch.py`) run the campaigns with the seeds of the protocols;
the three-optimizer campaigns and their amendment read their run lists (`wear-allowance/optimizers-*-v68.txt`) with
`xargs`, as their reports state.
Runtimes on one CPU core per run: about ninety seconds per wear-allowance PPO run at a quarter of the paper's budget
and several minutes at the full budget, an hour or more per fuel-tank controller, and about thirty-five minutes per
benchmark-suite run.

## Python environments

The frozen package lists are in [`environments/`](environments).

| Name | Packages | Used for |
|---|---|---|
| `constrained-rl` | numpy, scipy, matplotlib | every analysis and figure script |
| `constrained-rl-numerics` | torch 2.8, Stable-Baselines3 2.7, Gymnasium 1.2 with Box2D | the wear-allowance and fuel-tank campaigns |
| `constrained-rl-safety310` | Python 3.10, Safety-Gymnasium 1.0, MuJoCo 2.3.3 | the benchmark suite |

## Records and history

Protocols, run records, logs and audits are kept as they were written, including the revision names they carry.
[`ERRATA.md`](ERRATA.md) lists the corrections to the records (software versions, optimizer settings, the warm start's
optimizer state, the form of the parameter pull) with their evidence.
The protocols, which are records, call their statements predictions; the last section maps them to the paper's
observations.
[`PROVENANCE.md`](PROVENANCE.md) lists, for each protocol, its SHA-256, the date it states, and the private commits
in which the protocol and the campaign's first results first appear. `benchmark-suite/PROTOCOL-ramp-v65.md` and
`results-ramp-v65/` are a follow-up campaign (a ramped price and failure-time evaluation), mentioned but not reported
in the paper.

This repository is a curated copy of the numerics folder of the paper's working tree, taken on 2026-10-15; only the
paths to the toolbox and the output folder were changed, and wording in code comments. Earlier revisions of the paper
reported an emissions application, which revision 72 removed. Its folder is not part of this repository. The
three-optimizer and decoupled-pull protocols and reports of 2026-10-20 also describe runs on that application; those
records are not here either, and `analyze_optimizers_v68.py` skips them. Before publication, the home folder of the
authors' working tree was replaced by `<project>` in the run records and logs that recorded it (65 fuel-tank result
files and nine logs); nothing else in them changed.

## Licence and citation

The code is under the MIT licence ([`LICENSE`](LICENSE)); the protocols, reports, run records, trained policies,
tables and figures are under CC BY 4.0 ([`LICENSE-records.md`](LICENSE-records.md)). Cite the paper and this
repository as in [`CITATION.cff`](CITATION.cff). The repository is archived on Zenodo:
[10.5281/zenodo.23597426](https://doi.org/10.5281/zenodo.23597426) for all versions,
[10.5281/zenodo.23597427](https://doi.org/10.5281/zenodo.23597427) for the release of revision 73.

## Observations of the paper and the protocols

The paper reports its findings as observations, labelled (O1)–(O24) in the order of its experiments (the labels of
revision 73). An observation that a result of the paper implies names that result. Where the result's conditions
hold in the experiment, as for the bounds and rates computed on the exact benchmark and on the solvable flow, the
observation shows the slack of the bound. Where the result needs an assumption about the learner, such as a small
optimization error, the observation names that assumption. The table gives, for each observation, the protocol
statement that covered the same finding.

| Paper | Protocol file | Statement in the protocol |
|---|---|---|
| (O1) | `wear-allowance/PROTOCOL.md` | P1, the price–risk law |
| (O2) | `wear-allowance/PROTOCOL.md` | P1, the fall of risk with the price |
| (O3) | `wear-allowance/PROTOCOL.md` | P2 |
| (O4) | `wear-allowance/PROTOCOL.md` | P3 |
| (O5) | `wear-allowance/PROTOCOL.md` | P4, the utility cost of the barrier |
| (O6) | `wear-allowance/PROTOCOL.md` | P4, the effect of the barrier |
| (O7) | `wear-allowance/PROTOCOL.md` | P5 |
| (O8) | `fuel-tank/lunar-protocol.json` | none; the protocol holds no text of the statement |
| (O9) | `fuel-tank/lunar-protocol.json` | none; the protocol holds no text of the statement |
| (O10) | none (the solvable flow has no protocol file) | none |
| (O11) | none (the solvable flow has no protocol file) | none |
| (O12) | none (the solvable flow has no protocol file) | none |
| (O13) | `wear-allowance/PROTOCOL-anchor-v41.md` | (P12') and (P12'') |
| (O14) | `wear-allowance/PROTOCOL-anchor-v41.md` | (P13') |
| (O15) | none | none |
| (O16) | `wear-allowance/PROTOCOL-noise-v60.md` | (R1), the same quantity |
| (O17) | `wear-allowance/PROTOCOL-noise-v60.md` | (R2), the same quantity |
| (O18) | `wear-allowance/PROTOCOL-noise-v60.md` | (R3), the same quantity |
| (O19) | none | none |
| (O20) | `benchmark-suite/PROTOCOL-safety-v63.md` | (S1) |
| (O21) | `benchmark-suite/PROTOCOL-safety-v63.md` | (S4), the certificates |
| (O22) | `benchmark-suite/PROTOCOL-safety-v63.md` | (S2) |
| (O23) | `benchmark-suite/PROTOCOL-safety-v63.md` | (S5) |
| (O24) | `benchmark-suite/PROTOCOL-safety-v63.md` | (S4), the pass pattern |

Two statements whose premises did not arise appear in the paper as one unlabelled sentence each: the fuel-tank
statement on a tank the controller cannot respect, and (S3) of the benchmark-suite protocol. (R4) of
`wear-allowance/PROTOCOL-noise-v60.md` and (R1)–(R4) of `benchmark-suite/PROTOCOL-ramp-v65.md` do not appear in the
paper. The observations (O16)–(O18) report what the runs showed, not the protocol's statements.
