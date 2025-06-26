#!/bin/bash

OUTPUT="system_labels.csv"

echo "start,end,part" >> "$OUTPUT"

echo "Start logging"

./system_metrics_logging.sh &
METRICS_PID=$!  

echo "Logging started"

sleep 180

echo "Start probing part 1"

for ((i = 1; i <= 10; i++)); do
    echo "Iteration $i:"
    timestamp_start=$(date '+%Y-%m-%d %H:%M:%S.%N')
    sudo python3 probing.py -i 10 -n --description default
    timestamp_end=$(date '+%Y-%m-%d %H:%M:%S.%N')
    echo "$timestamp_start,$timestamp_end,1" >> "$OUTPUT"
    echo "Sleeping for 1 seconds..."
    sleep 10
done

echo "Probing part 1 finished"

sleep 180

echo "Start probing part 2"

for ((i = 1; i <= 10; i++)); do
    echo "Iteration $i:"
    timestamp_start=$(date '+%Y-%m-%d %H:%M:%S.%N')
    sudo python3 probing.py -n --description default
    timestamp_end=$(date '+%Y-%m-%d %H:%M:%S.%N')
    echo "$timestamp_start,$timestamp_end,2" >> "$OUTPUT"
    echo "Sleeping for 1 seconds..."
    sleep 10
done

echo "Probing part 2 finished"

sleep 180

kill $METRICS_PID
wait $METRICS_PID 2>/dev/null

echo "Done."
