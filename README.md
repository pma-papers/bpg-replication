# Replication of *The Price of Failure Risk*

This repository reproduces the numerical findings of the paper *The Price of Failure Risk: Clipped Barrier Policy
Gradients with Convex Regularization* (Malo, Viitasaari, Suominen, Vilkkumaa, Tahvonen). It holds every study's
simulator, training and analysis scripts, the protocols fixed before each campaign, and the records of every run, so
that each table and figure of the paper can be regenerated from the records, and each campaign can be run again.

It is separate from the toolbox. The toolbox, [`barrierpg`](https://github.com/pma-papers/barrierpg), is for applying
clipped barrier policy gradients to a new problem; this repository is for checking the paper. Results are named by the
titles the paper gives them, not by statement, section or table numbers, which change between revisions. The landing and wear tables were checked against revision 78 of the paper. The archived revision-77 release contains the reported experiment records. Revision 77 replaced the lander study by a landing
problem of our own (the runs of 17 and 18 November 2026, stages `formulations-*` and `ledger-*` in `fuel-tank/lunar/`),
added runs with regularization strengths lowered by a schedule or by a rule on the wear-allowance controller (stage
`schedules-v77`), and moved the benchmark-suite study out of the paper. The folders of the earlier lander studies and
of the benchmark suite stay here with their records.

The exploratory wear-allowance formulation comparison in Appendix E.1 is available with the
[toolbox code and seed-level results](https://github.com/pma-papers/barrierpg/blob/main/docs/wear-benchmark.md).
A separate [PPO architecture pilot](wear-allowance/ARCHITECTURE-PILOT.md) compares the published
network with two LayerNorm networks and includes its frozen source and all 24 run records.

## Studies

| Folder | Study in the paper | What it tests |
|---|---|---|
| [`wear-allowance/`](wear-allowance) | An asset operated under a wear allowance; the prior on a learned controller; noise and the anchor on a learned controller | The price against a grid-based benchmark, the adaptive multiplier, the interior barrier when the true allowance is smaller than the allowance used in training, the action anchor, the parameter anchor and parameter noise |
| [`fuel-tank/`](fuel-tank) | A learned lander paid for each landing (the landing problem: revenue 36 on the pad, fuel at 1 per unit, a tank of 24 units, built on the simulator of LunarLanderContinuous-v3); the fuel-tank studies of revisions up to 76 | The price on a learned controller and own-charge shortfalls against the best evaluated policies; in revisions up to 76, the dual step of an adaptive multiplier |
| [`solvable-flow/`](solvable-flow) | The convergence theorem and the regularizers on the analytical example | The slack of the curvature threshold, the prior policy, noise, drift and spread, the particle system |
| [`benchmark-suite/`](benchmark-suite) | A task of a benchmark suite, reported in revisions up to 76 and not in revision 77 | A Safety-Gymnasium navigation task: the price, the adaptive multiplier and the expected-cost formulation |
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
| Wear allowance: grid benchmark and campaign tables, the figure for a smaller true allowance | `wear-allowance/analyze_v41.py` | `constrained-rl` | `tables/wear-benchmark`, `tables/wear-campaign`, `figures/wear-margin` |
| Wear allowance: frontier, multiplier and anchor figures | `wear-allowance/relabel_wear_figures_v47.py` | `constrained-rl` | `figures/wear-frontier`, `wear-multiplier`, `parameter-anchor` |
| The prior on a learned controller: table | `wear-allowance/anchor_analyze_v41.py` | `constrained-rl` | `tables/parameter-anchor` |
| Roles of the regularizers on PPO: table (its figure left the manuscript on 2026-11-16) | `wear-allowance/analyze_noise_v60.py` | `constrained-rl` | `tables/wear-noise` |
| Roles of the regularizers on PPO, the anchor decoupled from Adam (2026-10-20, after revision 68) | `wear-allowance/analyze_decoupled_v68.py` | `constrained-rl` | no table of its own; `wear-allowance/REPORT-decoupled-v68.md` |
| Roles of the regularizers on PPO under three optimizers, with the amendment's rows (2026-10-20, after revision 68) | `wear-allowance/analyze_optimizers_v68.py` (per-coordinate records: `coordinate_diagnostic_v68.py`, `constrained-rl-numerics`) | `constrained-rl` | no table of its own; `wear-allowance/REPORT-optimizers-v68.md`, `wear-allowance/wear-optimizers-v68.tex` |
| Wear allowance, convergence and strengths changed during training (revision 76, 2026-11-14): runs with a trace after every rollout, a constant learning rate, anchor strengths changed in steps, noise removed, action regularization lowered, three fixed prices | `wear-allowance/wear_schedules_v76.py` (run lists `schedules-*-v76.txt`, runs in `wear-schedules-v76/campaign`) | `constrained-rl-numerics` | none directly |
| Wear allowance, the kernel of the action-regularized objective by dynamic programming | `wear-allowance/soft_benchmark_v76.py` | `constrained-rl` | `soft-benchmark-v76.json` |
| Wear allowance, the figures and tables on convergence, strength schedules and the three optimizers | `wear-allowance/regularization_figures_v76.py` (first drawings: `analyze_schedules_v76.py`, `campaign_convergence_v76.py`, `optimizers_figure_v76.py`) | `constrained-rl` | `figures/wear-optimizers` (paper), `figures/wear-anchor-traces`, `wear-anchor-schedules`, `wear-noise-traces`, `wear-action-kernel`, `wear-price-traces`, `tables/wear-optimizers`, `wear-anchor-schedules`, `wear-own-price` |
| Landing problem (revision 77): the figures and tables of the paper and of Online Resource 1, and every number of the text | `fuel-tank/lunar_landing_paper.py OUTDIR` | `constrained-rl` | `figures/landing-flights`, `landing-summary`, `landing-descent-seeds`, `landing-convergence`, `tables/landing`, `landing-values`, `landing-convergence`; `landing-numbers.json` |
| Landing problem: fuel before and after touchdown of the stored policies (500 episodes per policy) | `fuel-tank/lunar_touchdown_split.py` | `constrained-rl-numerics` | read by `lunar_landing_paper.py` (`lunar-touchdown-split-*.json`) |
| Landing problem: summaries of the three sets of runs, with training traces | `fuel-tank/lunar_ledger_main.py` | `constrained-rl` | none directly (`lunar-ledger-{main,staged,staged2}.*`) |
| Landing problem: the trial runs across formulations (benchmark reward with costly fuel, revenue on the pad; tanks of 18 to 24 units) | `fuel-tank/lunar_formulations_pilot.py` | `constrained-rl` | the table of trial runs in Online Resource 1 (`lunar-formulations-pilot*`) |
| Wear allowance, strengths lowered by a schedule against strengths held fixed (revision 77) | `wear-allowance/analyze_schedules_v77.py` (runs of `wear_schedules_v76.py --stage schedules-v77`, run list `schedules-v77-campaign.txt`, logs in `schedules-logs/`) | `constrained-rl` | `figures/wear-schedules`, `tables/wear-schedules` |
| Wear allowance, the pull lowered by a rule (revision 77) | `wear-allowance/analyze_rule_v77.py` (run list `schedules-v77-rule-campaign.txt`) | `constrained-rl` | `figures/wear-rule`, `tables/wear-rule` |
| Fuel tank (revisions up to 76): tables and figure | `fuel-tank/lunar_analyze_v41.py` | `constrained-rl` | `tables/fuel-tank`, `tables/fuel-tank-paired`, `tables/fuel-tank-price-risk`, `tables/fuel-tank-amount`, `figures/fuel-tank` |
| Fuel tank and landing problem: priced risk of the stored fixed-price policies (the failing frames of a second evaluation on the same episodes; `priced-risk.json` in each run folder) | `fuel-tank/lunar_reeval.py` | `constrained-rl-numerics` | read by `lunar_analyze_v41.py` |
| Solvable flow: convergence test | `solvable-flow/regularizer_figure_v76.py` (figures), `regularizer_figure_v46.py` (table) | `constrained-rl` | `tables/flow-test`, `figures/flow-test`, `figures/flow-test-below` |
| Solvable flow: prior strength | `solvable-flow/prior_figure.py` | `constrained-rl` | `tables/prior-sweep`, `figures/prior-sweep` |
| Solvable flow: regularization paths | `solvable-flow/prior_figure_v76.py`, `prior_path_table_v41.py` | `constrained-rl` | `figures/prior`, `tables/prior-paths` |
| Solvable flow: particle system | `solvable-flow/particle_flow_v58.py`, `particle_table_v58.py` | `constrained-rl` | `figures/particle-flow`, `tables/particle-flow` |
| Solvable flow: roles of the regularizers | `solvable-flow/roles_flow_v59.py`, `roles_table_v76.py` | `constrained-rl` | `figures/roles-flow`, `tables/roles-flow` |
| Solvable flow: a strength lowered in steps | `solvable-flow/decreasing_strength_v76.py` | `constrained-rl` | `figures/decreasing-strength`, `tables/decreasing-strength` |
| Solvable flow: the prior, the strength and a strength lowered in steps (figure of the paper) | `solvable-flow/prior_strength_figure_v76.py`, `unregularized_from_prior_v76.py` | `constrained-rl` | `figures/prior-strength` |
| Solvable flow: the runs from a Gaussian initial law | `solvable-flow/initial_law_check_v76.py` | `constrained-rl` | numbers of Appendix D.3 |
| Solvable flow: the bound of the corollary on the risk along the flow | `solvable-flow/corollary_bound_v60.py` | `constrained-rl` | `figures/corollary-bound` |
| Benchmark suite (revisions up to 76): table and figure | `benchmark-suite/analyze_safety_v63.py` (writes to `output/`) | `constrained-rl` | `tables/safety`, `figures/safety` |
| Illustrations of the theory (the logic of the clipped barrier, the identity on one path, the curvature budget along a geodesic of Gaussian laws) | `illustrations/introduction_schematic.py`, `running_example_figures.py`, `curvature_geodesic_v76.py` | `constrained-rl` | `figures/logic`, `identity`, `curvature` |
| Setup of the test problems (Online Resource 1): the wear-allowance problem | `wear-allowance/problem_figure_v76.py` | `constrained-rl` | `figures/wear-problem` |
| Setup of the test problems: the lander (flights near the pad in the design with costly fuel, stage `fuelprice-20-const`) | `fuel-tank/lunar_problem_eval_v76.py --stage fuelprice-20-const` (`constrained-rl-numerics`), then `lunar_paths_figure.py` | `constrained-rl` | `figures/fuel-tank-problem` |
| Setup of the test problems: fuel use in the stored fuel-tank study (tank of 20 units) | `fuel-tank/lunar_setup_eval.py` (`constrained-rl-numerics`), then `lunar_setup_figure_v76.py` | `constrained-rl` | `figures/fuel-tank-setup` |
| Setup of the test problems: the benchmark task | `benchmark-suite/safety_problem_eval_v76.py` (`constrained-rl-safety310`), then `safety_problem_figure_v76.py` | `constrained-rl` | `figures/benchmark-problem` |
| Setup of the analytical example, with its optimum in closed form | `solvable-flow/example_setup_v76.py` | `constrained-rl` | `figures/example-setup` |
| Illustrations on the wear-allowance example: the fixed-price proposition and the matched-risk theorem, the bounds on the failure time, the empirical certificate | `wear-allowance/price_geometry_v76.py`, `failure_time_tail_v76.py`, `validation_bounds_v76.py` | `constrained-rl` | `figures/price-geometry`, `failure-time-tail`, `validation-bounds` |
| Illustrations for Online Resource 1: the shadow price on an exact frontier, the consumption-based penalties, the admissible regularizers, the transfer to a network | `illustrations/shadow_price_frontier_v76.py`, `cvar_consumption_v76.py`, `regularizer_geometry_v76.py`, `network_transfer_v76.py` | `constrained-rl` | `figures/shadow-price-frontier`, `cvar-consumption`, `regularizer-geometry`, `network-transfer` |

`collect_figures.py <folder>` copies every figure above into a manuscript figure folder under the manuscript's
names. The figures are drawn at the journal's print width with sans-serif lettering (`lib/plotstyle.py`). The
generated tables reproduce the manuscript's table files of revision 76 byte for byte; the table of the roles of the
regularizers on the solvable flow, assembled by hand until revision 75, is now written by `roles_table_v76.py`. Three
scripts carry the corrections the paper made in revision 70: the stationary spread of injected noise under the anchor, without the value drift, is
`sqrt(kappa_1/kappa_2)` (`analyze_noise_v60.py`; the protocol's design values stay in its check (R1)), the
benchmark suite's validation shares the error `0.05` among all 72 reported candidates (`analyze_safety_v63.py`), and
the solvable flow's curvature constant is called a finite-difference estimate (`regularizer_figure_v41.py`).
The wear-allowance table's "bound est." column substitutes simulator estimates for the value and objective gap in the
price–risk expression. The binomial pass counts use the actual failure counts and their own confidence allocation. See `wear-allowance/analyze_v41.py`.

Four outputs are records that the paper no longer prints. The Online Resource was shortened on 9 November 2026 and
now points to this repository for them: the exact optima on the grid (`tables/wear-benchmark`), the runs over prior
locations and widths (`tables/prior-sweep`, `figures/prior-sweep`), the regularization paths as a table
(`tables/prior-paths`) and the bound of the corollary along the flow (`figures/corollary-bound`). The figure of the
check of noise and drift on the analytical example (`figures/roles-flow`) is printed in the Online Resource since the same day. The same revision
writes the parameter of a unit as `x`, the anchor as `x_0` and the feature of the solvable flow as `varphi`, and the
labels of the tables and figures follow. Since the night of 9 November the paper calls the solvable flow the
*analytical example*; the folder `solvable-flow/` and the script
names keep the old name.

## Running the campaigns again

The launch scripts (`*campaign*.sh`, `*run_one*.sh`, `launch.py`) run the campaigns with the seeds of the protocols;
the three-optimizer campaigns and their amendment read their run lists (`wear-allowance/optimizers-*-v68.txt`) with
`xargs`, as their reports state.
Runtimes on one CPU core per run: about ninety seconds per wear-allowance PPO run at a quarter of the paper's budget
and several minutes at the full budget, an hour or more per fuel-tank controller, and about thirty-five minutes per
benchmark-suite run. The runs of the landing problem and of the schedules were run on the CSC cluster Roihu with the
job scripts in `fuel-tank/roihu/` and their run lists (`runs-formulations*.txt`, `runs-ledger.txt`); a run of the
landing problem takes about fifty minutes of 20 million steps there.

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
optimizer state, the form of the parameter anchor) with their evidence.
The protocols, which are records, call their statements predictions; the last section maps them to the paper's
observations.
[`PROVENANCE.md`](PROVENANCE.md) lists, for each protocol, its SHA-256, the date it states, and the private commits
in which the protocol and the campaign's first results first appear. `benchmark-suite/PROTOCOL-ramp-v65.md` and
`results-ramp-v65/` are a follow-up campaign (a ramped price and failure-time evaluation), mentioned but not reported
in the paper.

This repository is a curated copy of the numerics folder of the paper's working tree, taken on 2026-10-15; only the
paths to the toolbox and the output folder were changed, and wording in code comments. Earlier revisions of the paper
reported an emissions application, which revision 72 removed. Its folder is not part of this repository. The
three-optimizer and decoupled-anchor protocols and reports of 2026-10-20 also describe runs on that application; those
records are not here either, and `analyze_optimizers_v68.py` skips them. Before publication, the home folder of the
authors' working tree was replaced by `<project>` in the run records and logs that recorded it (65 fuel-tank result
files and nine logs); nothing else in them changed. The same replacement was made in the fifty result files and the log
of the fuel-tank runs added on 2026-11-10, and in the records of the landing problem and of the schedules, run on
the CSC cluster Roihu from 13 to 18 November 2026, where the CSC project and user name in the recorded paths were
also replaced.

## Licence and citation

The code is under the MIT licence ([`LICENSE`](LICENSE)); the protocols, reports, run records, trained policies,
tables and figures are under CC BY 4.0 ([`LICENSE-records.md`](LICENSE-records.md)). Cite the paper and this
repository as in [`CITATION.cff`](CITATION.cff). The repository is archived on Zenodo:
[10.5281/zenodo.23597426](https://doi.org/10.5281/zenodo.23597426) for all versions,
[10.5281/zenodo.24369373](https://doi.org/10.5281/zenodo.24369373) for the release of revision 77,
[10.5281/zenodo.24194766](https://doi.org/10.5281/zenodo.24194766) for the release of revision 76,
[10.5281/zenodo.23986660](https://doi.org/10.5281/zenodo.23986660) for the release of revision 75, and
[10.5281/zenodo.23597427](https://doi.org/10.5281/zenodo.23597427) for the release of revision 73, whose records and numbers are the same.

## Observations of the paper and the protocols

The paper reports its findings as observations, labelled (O1)–(O12) in the paper and (O13)–(O24) in Online Resource 1,
in the order of its experiments. These are the labels of revision 77. Against revision 76, which had (O1)–(O26),
revision 77 states the former (O4) in a sentence without a label, so the former (O5)–(O7) are (O4)–(O6). (O7) and
(O8) concern the landing problem, which replaced the earlier lander study, and the former observation on the adaptive
multiplier was dropped. The former (O10)–(O17) are (O9)–(O16). The former (O18)–(O21), on noise and the anchor, are
(O17)–(O20), and (O21)–(O24) are new and concern strengths lowered by a schedule or by a rule. The former (O22)–(O26),
on the benchmark-suite study, are not in revision 77. An observation that a result of the paper implies names that
result. The analytical example compares observed rates with the theoretical rate where its hypotheses hold. The wear
benchmark uses a grid and continuous-simulator evaluation. Its measured shortfalls do not certify continuous-model optimization gaps. The table
gives, for each observation, the protocol statement that covered the same finding.

| Paper | Protocol file | Statement in the protocol |
|---|---|---|
| (O1) | `wear-allowance/PROTOCOL.md` | P1, the price–risk bound |
| (O2) | `wear-allowance/PROTOCOL.md` | P1, the fall of risk with the price |
| (O3) | `wear-allowance/PROTOCOL.md` | P2 |
| (O4) | `wear-allowance/PROTOCOL.md` | P4, the utility cost of the barrier |
| (O5) | `wear-allowance/PROTOCOL.md` | P4, the effect of the barrier |
| (O6) | `wear-allowance/PROTOCOL.md` | P5 |
| (O7) | `fuel-tank/lunar-protocol.json`, stages `ledger-*` | none; the stages record the design, and the entry of the stages `ledger-staged-*` the reason for the start from the controller trained without a price |
| (O8) | the same stages | none; formulated after the runs |
| (O9) | none (the analytical example has no protocol file) | none |
| (O10) | none (the analytical example has no protocol file) | none |
| (O11) | none (the analytical example has no protocol file) | none |
| (O12) | none (the analytical example has no protocol file) | none; formulated after the runs |
| (O13) | none | none |
| (O14) | `wear-allowance/PROTOCOL-anchor-v41.md` | (P12') and (P12'') |
| (O15) | `wear-allowance/PROTOCOL-anchor-v41.md` | (P13'), for the anchor on every parameter; formulated after the runs for the anchor on the mean network only |
| (O16) | none (the runs of `wear-allowance/wear_schedules_v76.py`) | none; formulated after the runs |
| (O17) | `wear-allowance/PROTOCOL-noise-v60.md`, `wear-allowance/PROTOCOL-optimizers-v68.md`, `wear-allowance/PROTOCOL-decoupled-v68.md` | (R1), (E2)–(E4), (D1) and (D2), the same quantity |
| (O18) | the same three files | (R2), (E5) and (D4), the same quantity |
| (O19) | `wear-allowance/PROTOCOL-optimizers-v68.md`, `wear-allowance/PROTOCOL-decoupled-v68.md` | (E5), reported without a numerical statement, and (D3), which concerns the spread in the same runs |
| (O20) | `wear-allowance/PROTOCOL-noise-v60.md` | (R3), the same quantity; none for the runs with the noise lowered during training (`wear-allowance/wear_schedules_v76.py`) |
| (O21) | none (stage `schedules-v77`) | none; formulated after the runs |
| (O22) | none (stage `schedules-v77`) | none; formulated after the runs |
| (O23) | none (stage `schedules-v77`) | none; formulated after the runs |
| (O24) | none (stage `schedules-v77`, runs with the rule) | none; formulated after the runs |

The statement (P3) of `wear-allowance/PROTOCOL.md` appears in the paper as one sentence without a label. Two
statements whose premises did not arise appear in revision 76 as one unlabelled sentence each: the fuel-tank
statement on a tank the controller cannot respect, and (S3) of the benchmark-suite protocol. (R4) of
`wear-allowance/PROTOCOL-noise-v60.md` and (R1)–(R4) of `benchmark-suite/PROTOCOL-ramp-v65.md` do not appear in the
paper. The paper says that all observations of Appendix D.6 of Online Resource 1, (O17)–(O24), were formulated after
the runs; (O17)–(O20) report what the runs showed, not the protocol's statements. The protocol and report files keep
the observation labels of the revision in which they were written.
