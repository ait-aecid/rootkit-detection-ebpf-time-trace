#!/bin/bash

set -euo pipefail

scp user.py kernel.c data_classes.py ubuntu2204:

ssh ubuntu2204 "sudo python3 user.py -r 100 -e ./getpid_opendir_readdir_root"

newest_output=$(ssh ubuntu2204 "ls -1 output*" | tail -n 1)

scp ubuntu2204:"$newest_output" ./

python run.py "$newest_output"
