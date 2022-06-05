#!/bin/bash

python3 -m pip install --user pipenv
export PATH=$(python3 -m site --user-base)/bin:$PATH
pipenv install