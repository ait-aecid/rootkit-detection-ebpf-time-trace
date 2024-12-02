#!/bin/bash

set -euo pipefail

TARGET=ubuntu2204_linux6.x

scp probing.py kernel.c data_classes.py linux.py $TARGET:

ssh $TARGET "sudo python3 probing.py --rootkit --normal --drop-boundary-events --visible-files 100 --hidden-files 100 -i 100 -e ./open_getdents"

scp -r $TARGET:"events/" ./

source venv3.9/bin/activate
python run.py
