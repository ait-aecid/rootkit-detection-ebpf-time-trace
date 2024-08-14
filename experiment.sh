#!/bin/bash

set -euo pipefail

TARGET=ubuntu2204_linux6.x

scp user.py kernel.c data_classes.py linux.py $TARGET:

ssh $TARGET "sudo python3 user.py -i 100 -e ./open_getdents"

newest_output=$(ssh $TARGET "ls -1 output*" | tail -n 1)

scp $TARGET:"$newest_output" ./

source venv/bin/activate
python run.py "$newest_output"
