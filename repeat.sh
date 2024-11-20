#!/bin/bash

# Loop 100 times
for i in {1..100}; do
    echo "Iteration $i:"
    sudo python3 probing.py -rn
    echo "Sleeping for 10 seconds..."
    sleep 10
done

echo "All iterations completed."
