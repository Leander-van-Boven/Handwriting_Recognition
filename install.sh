#!/bin/bash

git submodule init
git submodule update
pipenv install 
pipenv shell
