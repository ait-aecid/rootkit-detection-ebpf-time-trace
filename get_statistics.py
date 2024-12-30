import array
import json
from itertools import chain
from statistics import median
from typing import *

from multiprocessing import Process
import argparse
import sys
import os
import re
import gzip
import math
import time
import random
import json
import numpy as np
import pandas as pd
from data_classes import Event, Interval, experiment_from_json
from scipy.stats import ttest_ind
from scipy.stats import norm
from scipy.stats import chi2
from tqdm import tqdm
import psutil

normal_key = "normal"
rootkit_key = "rootkit"

parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", default="events", type=str, help="Directory containing event data.")

args = parser.parse_args()

if not os.path.isdir(args.directory):
    print("Error: " + args.directory + " is not a valid directory.")
    exit()

start_times = {}
end_times = {}
num_events = {}
num_batches = {}
num_intervals = {}
num_functions = {}
num_function_pairs = {}
num_pids = {}
files = os.listdir(args.directory)
files.sort()
for filename in tqdm(files):
    filepath = os.path.join(args.directory, filename)
    
    # Check if it's a file (not a directory or symbolic link)
    if os.path.isfile(filepath):
        experiment = None
        with gzip.open(filepath, 'r') as file:
            json_obj = json.load(file)
            experiment = experiment_from_json(json_obj)
        if experiment.description not in num_events:
            num_events[experiment.description] = {}
            start_times[experiment.description] = {}
            end_times[experiment.description] = {}
            num_batches[experiment.description] = {}
            num_intervals[experiment.description] = {}
            num_functions[experiment.description] = {}
            num_pids[experiment.description] = {}
            num_function_pairs[experiment.description] = {}
        if experiment.label not in num_events[experiment.description]:
            num_events[experiment.description][experiment.label] = []
            start_times[experiment.description][experiment.label] = []
            end_times[experiment.description][experiment.label] = []
            num_batches[experiment.description][experiment.label] = 0
            num_intervals[experiment.description][experiment.label] = []
            num_functions[experiment.description][experiment.label] = []
            num_pids[experiment.description][experiment.label] = []
            num_function_pairs[experiment.description][experiment.label] = []
        file_date = filename[(filename.find("T") + 1):filename.find(".")].replace("_", ":") #.replace("events/", "").replace("experiment", "").replace(".json", "").replace(".gz", "")
        if len(start_times[experiment.description][experiment.label]) == 0:
            start_times[experiment.description][experiment.label] = file_date
        end_times[experiment.description][experiment.label] = file_date
        num_events[experiment.description][experiment.label].append(len(experiment.events))
        num_batches[experiment.description][experiment.label] += 1
        processes = {}
        for event in experiment.events:
            try:
                processes[event.pid]
            except KeyError:
                processes[event.pid] = []
            processes[event.pid].append(event)
        num_pids[experiment.description][experiment.label].append(len(processes))

        functions = set()
        function_pairs = set()
        intervals_cnt = 0
        for pid in processes:
            for i in range(len(processes[pid]) - 1):
                event_a = processes[pid][i]
                event_b = processes[pid][i + 1]
                type_name = event_a.probe_point + ":" + event_b.probe_point
                #try:
                #    intervals_time[type_name]
                #except KeyError:
                #    intervals_time[type_name] = []
                #intervals_time[type_name].append(event_b.timestamp - event_a.timestamp)
                functions.add(event_a.probe_point)
                functions.add(event_b.probe_point)
                function_pairs.add(type_name)
                intervals_cnt += 1
        num_intervals[experiment.description][experiment.label].append(intervals_cnt)
        num_functions[experiment.description][experiment.label].append(len(functions))
        num_function_pairs[experiment.description][experiment.label].append(len(function_pairs))
for description, d in num_events.items():
    for label, res in d.items():
        num_events_per_batch = np.array(num_events[description][label]) / num_batches[description][label]
        num_intervals_per_batch = np.array(num_intervals[description][label]) / num_batches[description][label] # This is almost the same as num_events_per_batch, because all pairs of events are considered (i.e., num_events - 1 intervals for num_events events)
        s = description + " & " 
        s += label + " & " 
        s += start_times[description][label] + " & " 
        s += end_times[description][label] + " & "
        s += str(num_batches[description][label]) + " & " 
        #s += str(round(np.mean(num_events_per_batch), 2)) + ", " + str(round(np.var(num_events_per_batch), 2)) + " & " 
        s += str(round(np.median(num_events_per_batch), 2))
        #s += str(np.mean(num_intervals_per_batch)) + ", " + str(np.var(num_intervals_per_batch)) + " & "
        #s += str(np.mean(num_functions[description][label])) + " & " 
        #s += str(round(np.mean(num_function_pairs[description][label]), 2))  + " & "
        #s += str(np.mean(num_pids[description][label]))
        print(s)


