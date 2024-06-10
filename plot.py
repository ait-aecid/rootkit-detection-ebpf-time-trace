import json
import matplotlib.pyplot as plt
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
    type_name = events[i].probe_point + ":" + events[i+1].probe_point
    try:
        intervals[type_name]
    except KeyError:
        intervals[type_name] = []
    intervals[type_name].append(events[i+1].timestamp - events[i].timestamp)

def unique_vals(lst: list) -> int:
    import numpy as np
    unique_values, value_counts = np.unique(lst, return_counts=True)
    return len(unique_values)

for name, values in intervals.items():
    print(name + "...")
    plt.hist(values, bins=unique_vals(values))
    plt.title(name)
    plt.ticklabel_format(style='plain', axis='both')
    plt.savefig("distribution_" + file_date + "_" + name + ".svg")
