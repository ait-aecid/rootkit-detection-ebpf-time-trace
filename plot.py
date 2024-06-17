import json
import matplotlib.pyplot as plt
from multiprocessing import Process
import os
import re
import sys
from data_classes import Event

try:
    filename = sys.argv[1]
except IndexError:
    # get the most recent output file...
    files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
    pattern = r'^output.*\.json$'
    outputs = [file for file in files if re.match(pattern, file)]
    filename = outputs[-1]

with open(filename, 'r') as file:
    json_obj = json.load(file)
    events = [Event(**elem) for elem in json_obj]

file_date = filename.replace("output", "").replace(".json", "")

intervals = {}
for i in range(len(events) - 1):
    event_a = events[i]
    event_b = events[i+1]
    if event_a.pid == event_b.pid and event_a.tgid == event_b.tgid:
        type_name = event_a.probe_point + ":" + event_b.probe_point
        try:
            intervals[type_name]
        except KeyError:
            intervals[type_name] = []
        intervals[type_name].append(event_b.timestamp - event_a.timestamp)


def unique_vals(lst: list) -> int:
    values = []
    unique = 0
    for elem in lst:
        if elem not in values:
            values.append(elem)
            unique += 1
    return unique


def make_histogram(name: str, values: [int]) -> None:
    plt.hist(values, bins=unique_vals(values))
    plt.title(name)
    plt.yscale('log')
    plt.savefig("distribution_" + file_date + "_" + name + ".svg")
    print(name + " saved")


plot_processes = []

for name, values in intervals.items():
    worker = Process(target=make_histogram, args=[name, values])
    worker.start()
    plot_processes.append(worker)

for worker in plot_processes:
    worker.join()
