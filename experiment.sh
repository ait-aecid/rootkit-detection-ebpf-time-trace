#!/bin/bash

set -euo pipefail

TARGET=ubuntu2204_linux6.x

scp detection.py kernel.c data_classes.py linux.py $TARGET:

ssh $TARGET "sudo python3 detection.py -rn -i 100 -e ./open_getdents"

newest_experiment=$(ssh $TARGET "ls -1 experiment*" | tail -n 1)

scp $TARGET:"$newest_experiment" ./

source venv3.9/bin/activate
python run.py "$newest_experiment"
