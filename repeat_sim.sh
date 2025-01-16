#!/bin/bash

# Default: Loop normal 10 times
for i in {1..10}; do
    echo "Iteration $i:"
    sudo python3 probing.py -n --description default
    echo "Sleeping for 10 seconds..."
    sleep 10
done

# Default: Loop normal and rootkit 100 times
for i in {1..100}; do
    echo "Iteration $i:"
    sudo python3 probing.py -nr --description default
    echo "Sleeping for 10 seconds..."
    sleep 10
done

# Random file count in range 10-100: Loop normal and rootkit 100 times
for i in {1..100}; do
    echo "Iteration $i:"
    visible_files=$((10 + RANDOM % 91))
    hidden_files=$((10 + RANDOM % 91))
    sudo python3 probing.py -nr --visible-files $visible_files --hidden-files $hidden_files --description file_count
    echo "Sleeping for 10 seconds..."
    sleep 10
done

# System load: Loop normal and rootkit 100 times
for i in {1..100}; do
    echo "Iteration $i:"
    sudo python3 probing.py -nr --load --description system_load
    echo "Sleeping for 10 seconds..."
    sleep 10
done

# Execute ls-basic: Loop normal and rootkit 100 times
for i in {1..100}; do
    echo "Iteration $i:"
    sudo python3 probing.py -nr --executable ./ls-basic --description ls_basic
    echo "Sleeping for 10 seconds..."
    sleep 10
done

echo "All iterations completed."
