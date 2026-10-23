#!/bin/sh
# usage: run_one_v65.sh train KIND M SEED  |  run_one_v65.sh reeval RUN_DIR      (protocol PROTOCOL-ramp-v65.md)
cd "$(dirname "$0")"
PY="$HOME/.venvs/constrained-rl-safety310/bin/python"
mkdir -p logs
case $1 in
  train)  PYTHONDONTWRITEBYTECODE=1 "$PY" safety_ramp_v65.py train --kind "$2" --m "$3" --seed "$4" >> "logs/v65_train_$2_$3_$4.log" 2>&1;;
  reeval) PYTHONDONTWRITEBYTECODE=1 "$PY" safety_ramp_v65.py reeval --run "$2" >> logs/v65_reeval.log 2>&1;;
esac
