# PPO architecture pilot for the wear allowance

This benchmark asks whether residual connections improve learning under a fixed failure price. It keeps
the simulator, observations, action scaling, failure-only reward and PPO settings of the published
wear campaign. The price is `m=1`, the wear shock has shape `0.5`, and the training budget is eight
million environment steps before rounding up to a complete rollout.

Each of eight training seeds is used with three architectures.

| Architecture | Actor and critic |
| --- | --- |
| `published` | The campaign's two 64-unit tanh layers |
| `normalized` | A 32-unit projection and two pre-LayerNorm blocks with SiLU and no skips |
| `residual` | The same normalized architecture with skips in both networks |

The comparison of `residual` with `normalized` holds the parameter count, initial parameter values
and activation fixed. Both have 4,482 actor and 4,481 critic parameters. The comparison with
`published` measures the combined architecture change. Start with wear and use the paired learning
curves to decide whether a landing study is warranted.

## Outcomes

At the first complete rollout at or after each budget (32,768, 500,000, one million, two million,
four million and eight million steps), evaluate the stochastic policy on a fresh batch. Training
uses one uninterrupted learning-rate schedule. The rollout probes use 4,000 episodes each and the
final policy uses 20,000. Compare paired stopped utility $U$, breakdown risk $q_\beta$ and priced value
$U-20q_\beta$ versus environment interactions and training wall time (excluding evaluation). Evaluation seeds are paired
across architectures and distinct across checkpoints. Every final candidate is evaluated regardless
of the probe results.

For each of the 24 final candidates, check whether a one-sided exact binomial upper bound for the
probability of breakdown before maintenance is at most 5%. Allocate a family error of 5% over the
24 checks. A candidate can pass the risk check even when another architecture reaches higher utility.
A candidate whose upper bound exceeds 5% has insufficient evidence to pass this check. This tolerance
applies to the wear benchmark. Forestry application requires evaluation under its own empirical constraints.

## Results from the 24 runs

Values are means over eight paired seeds at 8,003,584 environment steps. Training times exclude
evaluation and refer to this one-core Roihu run. All 24 policies passed the final risk check. The
largest family-adjusted one-sided upper bound was 2.96%.

| Architecture | Stopped utility | Breakdown risk | Priced value | Training time |
| --- | ---: | ---: | ---: | ---: |
| Published tanh MLP | 4.492 | 1.83% | 4.125 | 9.2 min |
| LayerNorm without skips | 4.491 | 2.13% | 4.065 | 15.0 min |
| LayerNorm with skips | 4.499 | 2.18% | 4.063 | 15.8 min |

The published architecture had the highest mean priced value at all six checkpoints. The two
LayerNorm architectures gave similar final priced values. This comparison holds the failure
price and the PPO settings fixed. [Per-seed records](architecture-pilot/results.json)

## Running on Roihu

Use the [frozen source archive](architecture-pilot/sources.tar.gz) to rerun the study. It contains
the exact Python sources used in the pilot, including the policy implementation. The recorded CPU
environment used PyTorch 2.8.0, Stable-Baselines3 2.7.1, Gymnasium 1.2.3 and NumPy 2.2.6.
For a local smoke test, extract the archive into a new directory and run
`python benchmark_architectures.py --task 2 --smoke --output smoke` from that directory.

For Roihu, stage the archive and `run_architectures_roihu.sh` in the project's scratch directory.
On the Roihu CPU login node, set `CSC_PROJECT` to the project and change to the staged directory.
The script uses compute-node local storage and copies one trained model and one result record per
completed task to project scratch. Set `python_path` to your CPU Python environment.

```bash
python_path="/projappl/${CSC_PROJECT}/${USER}-bpg/numerics313/bin/python"
mkdir -p results logs
sbatch --account="$CSC_PROJECT" --partition=test --array=2 --time=00:15:00 \
    --output=logs/smoke-%A_%a.out run_architectures_roihu.sh \
    "$PWD/sources.tar.gz" "$PWD/results" "$python_path" --smoke
```

Check the smoke job's output and memory before submitting the array with the same bundle and
environment. The smoke run exercises the residual architecture and artifact staging with two rollouts
and 64 evaluation episodes per checkpoint. It makes no risk certification.

```bash
sbatch --account="$CSC_PROJECT" --output=logs/pilot-%A_%a.out run_architectures_roihu.sh \
    "$PWD/sources.tar.gz" "$PWD/results" "$python_path"
```

The array caps concurrency at eight. At the requested limits of one core, 2 GiB and 30 minutes per
task, the 24 tasks account for up to 9 CPU billing units, plus up to 0.1875 CPU billing units for the
15-minute smoke job and any storage charges. [CSC billing](https://docs.csc.fi/supercomputing/hpc-billing/)
