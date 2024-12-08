#!/bin/bash

sleep_time=10
normal_count=150
rootkit_count=100

# Default: Loop normal 150 times
for i in {1..$normal_count}; do
    echo "Iteration $i:"
    sudo python3 probing.py -n --description default
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

# Default: Loop rootkit 100 times
for i in {1..$rootkit_count}; do
    echo "Iteration $i:"
    sudo python3 probing.py -r --description default
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

# Random file count in range 10-100: Loop normal 150 times
for i in {1..$normal_count}; do
    echo "Iteration $i:"
    visible_files=$((10 + RANDOM % 91))
    hidden_files=$((10 + RANDOM % 91))
    sudo python3 probing.py -n --visible-files $visible_files --hidden-files $hidden_files --description file_count
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

# Random file count in range 10-100: Loop rootkit 100 times
for i in {1..$rootkit_count}; do
    echo "Iteration $i:"
    visible_files=$((10 + RANDOM % 91))
    hidden_files=$((10 + RANDOM % 91))
    sudo python3 probing.py -r --visible-files $visible_files --hidden-files $hidden_files --description file_count
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

# System load: Loop normal and rootkit 150 times
for i in {1..$normal_count}; do
    echo "Iteration $i:"
    sudo python3 probing.py -n --load --description system_load
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

# System load: Loop normal and rootkit 100 times
for i in {1..$rootkit_count}; do
    echo "Iteration $i:"
    sudo python3 probing.py -r --load --description system_load
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

# Execute ls-basic: Loop normal 150 times
for i in {1..$normal_count}; do
    echo "Iteration $i:"
    sudo python3 probing.py -n --executable ./ls-basic --description ls_basic
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

# Execute ls-basic: Loop rootkit 100 times
for i in {1..$rootkit_count}; do
    echo "Iteration $i:"
    sudo python3 probing.py -r --executable ./ls-basic --description ls_basic
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

# Execute filename length: Loop normal 150 times
for i in {1..$normal_count}; do
    echo "Iteration $i:"
    file_name_length=$((20 + RANDOM % 41))
    sudo python3 probing.py -n --file-name-length $file_name_length --description filename_length
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

# Execute filename length: Loop rootkit 100 times
for i in {1..$rootkit_count}; do
    echo "Iteration $i:"
    file_name_length=$((20 + RANDOM % 41))
    sudo python3 probing.py -r --file-name-length $file_name_length --description filename_length
    echo "Sleeping for $sleep_time seconds..."
    sleep $sleep_time
done

echo "All iterations completed."
