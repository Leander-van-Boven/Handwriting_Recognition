#!/bin/bash

export PIPENV_PIPFILE="../Pipfile"
pipenv shell
python3 ../main.py -outdir . dss --indir $1 --glob '*'