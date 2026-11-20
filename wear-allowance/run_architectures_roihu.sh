#!/bin/bash
#SBATCH --job-name=wear-architectures
#SBATCH --partition=small
#SBATCH --array=0-23%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ( $# -eq 4 && $4 != --smoke ) ]]; then
    echo "Usage: sbatch $0 SOURCE_TAR OUTPUT_ROOT PYTHON [--smoke]" >&2
    exit 2
fi

source_tar=$1
output_root=$2
python=$3
task=$SLURM_ARRAY_TASK_ID
job_tag=${SLURM_ARRAY_JOB_ID}_${task}
stage=$(mktemp -d "${TMPDIR:?Roihu must provide compute-node local storage}/bpg-wear.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+$APPTAINER_BINDPATH,}$stage"
export MPLCONFIGDIR="$stage/matplotlib" XDG_CACHE_HOME="$stage/cache" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

sha256sum "$source_tar"
tar -xf "$source_tar" -C "$stage"
cd "$stage"
srun "$python" benchmark_architectures.py --task "$task" --output "$stage/result" "${@:4}"

# The result record is published last and indicates that the model was copied successfully.
destination=$output_root/$job_tag
mkdir "$destination"
cp "$stage/result/final_model.zip" "$destination/"
cp "$stage/result/result.json" "$destination/result.json.tmp"
mv "$destination/result.json.tmp" "$destination/result.json"
