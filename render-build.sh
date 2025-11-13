#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset

echo "---- Forçage de la version Python 3.10 ----"
echo "python-3.10.13" > runtime.txt

echo "---- Mise à jour de pip ----"
python -m pip install --upgrade pip

echo "---- Installation des dépendances ----"
pip install -r requirements.txt
