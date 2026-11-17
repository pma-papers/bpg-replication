#!/bin/bash
# The pinned packages of environments/constrained-rl-numerics.txt, installed as listed (no dependency resolution:
# the list is complete), torch in its CPU build. Box2D is built from its source with the swig of the base.
set -e
pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu \
Box2D==2.3.10 \
cloudpickle==3.1.2 \
contourpy==1.4.0 \
cycler==0.12.1 \
Farama-Notifications==0.0.6 \
filelock==4.0.9 \
fonttools==4.66.1 \
fsspec==2026.9.0 \
gymnasium==1.2.3 \
Jinja2==3.1.6 \
kiwisolver==1.5.1 \
MarkupSafe==3.0.3 \
matplotlib==3.11.2 \
mpmath==1.3.0 \
networkx==3.7 \
numpy==2.2.6 \
packaging==26.3 \
pandas==3.0.6 \
pillow==12.3.0 \
pygame==2.6.1 \
pyparsing==3.3.3 \
python-dateutil==2.9.0.post0 \
scipy==1.15.3 \
setuptools==84.0.0 \
six==1.17.0 \
stable_baselines3==2.7.1 \
sympy==1.14.0 \
torch==2.8.0 \
typing_extensions==4.16.0
