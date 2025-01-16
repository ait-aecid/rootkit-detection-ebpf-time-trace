import os
import json
import gzip
from tqdm import tqdm
from data_classes import Event, Interval, experiment_from_json

files = os.listdir("events")
files.sort()
print("Found " + str(len(files)) + " files")
counts = {}
total_num_events = 0
with open("event_counts.csv", "w+") as out:
    out.write("filename,name,description,label,count\n")
    for filename in tqdm(files):
        filepath = os.path.join("events", filename)
        if filename not in counts:
            counts[filename] = {}
        with gzip.open(filepath, 'r') as file:
            json_obj = json.load(file)
            experiment = experiment_from_json(json_obj)
            if experiment.label not in counts[filename]:
                counts[filename][experiment.label] = {}
            if experiment.description not in counts[filename][experiment.label]:
                counts[filename][experiment.label][experiment.description] = {}
            for event in experiment.events:
                total_num_events += 1
                if event.probe_point not in counts[filename][experiment.label][experiment.description]:
                    counts[filename][experiment.label][experiment.description][event.probe_point] = 1
                else:
                    counts[filename][experiment.label][experiment.description][event.probe_point] += 1
    for fn, fn_dict in counts.items():
        for label, label_dict in fn_dict.items():
            for description, description_dict in label_dict.items():
                for probe_point, count in description_dict.items():
                    out.write(fn + "," + probe_point + "," + description + "," + label + "," + str(count) + "\n")
print("Total number of events: " + str(total_num_events))
