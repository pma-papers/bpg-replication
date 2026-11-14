#!/bin/sh
# Runs the lines of a campaign list, at most $2 runs of wear_schedules_v76.py at a time (counted on the whole machine).
cd "$(dirname "$0")" || exit 1
while read -r line; do
  while [ "$(pgrep -f wear_schedules_v76.py | wc -l)" -ge "${2:-8}" ]; do sleep 5; done
  ./schedules_run_one_v76.sh $line &
  sleep 3
done < "$1"
wait
