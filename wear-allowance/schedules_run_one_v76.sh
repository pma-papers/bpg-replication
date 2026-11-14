#!/bin/sh
# One run of wear_schedules_v76.py; the arguments are one line of a campaign list. Log in schedules-logs/.
cd "$(dirname "$0")" || exit 1
name=$(echo "$*" | tr -c 'A-Za-z0-9.@\n' '_')
exec "$HOME/.venvs/constrained-rl-numerics/bin/python" wear_schedules_v76.py "$@" >> "schedules-logs/$name.log" 2>&1
