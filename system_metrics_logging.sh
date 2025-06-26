#!/bin/bash

OUTPUT="system_metrics.csv"
DEVICE="vda"
SECTOR_SIZE=512
INTERVAL=1  # in seconds

# Print CSV header
echo "timestamp,cpu_usage_percent,mem_usage_percent,disk_read_kB,disk_write_kB" > "$OUTPUT"

while true; do
    read r1 w1 <<< $(awk -v dev="$DEVICE" '$3 == dev {print $6, $10}' /proc/diskstats)

    timestamp=$(date '+%Y-%m-%d %H:%M:%S.%N')

    # CPU usage (%idle to usage)
    cpu_idle=$(mpstat 1 1 | awk '/Average:/ {print $12}')
    cpu_usage=$(echo "scale=2; 100 - $cpu_idle" | bc)

    # Memory usage (percentage)
    read mem_total mem_used <<< $(free | awk '/Mem:/ {print $2, $3}')
    mem_usage_percent=$(echo "scale=2; ($mem_used / $mem_total) * 100" | bc)

    # Disk I/O (read/write in kB/s)
    #read d_read d_write <<< $(iostat -d -y 1 2 | awk '/vda/ {print $3, $4}')

    sleep "$INTERVAL"

    read r2 w2 <<< $(awk -v dev="$DEVICE" '$3 == dev {print $6, $10}' /proc/diskstats)
    read_kb=$(( (r2 - r1) * SECTOR_SIZE / 1024 ))
    write_kb=$(( (w2 - w1) * SECTOR_SIZE / 1024 ))

    # Log to CSV
    echo "$timestamp,$cpu_usage,$mem_usage_percent,$read_kb,$write_kb" >> "$OUTPUT"

    #sleep "$INTERVAL"
done
