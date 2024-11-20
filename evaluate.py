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
import numpy as np
import pandas as pd
import pandasql as psql
from data_classes import Event, Interval, experiment_from_json

class Intervals:
    def __init__(self, filename):
        self.args = []
        self.processes = {}
        self.intervals = {}
        self.interval_types = []
        self.file_date = ""
        self.label = None
        self.experiment = None
        self.events = []
        self.events_per_process = {}
        self.dataframe = None
        self.event_count = {}

        with gzip.open(filename, 'r') as file:
            json_obj = json.load(file)
            experiment = experiment_from_json(json_obj)
            self.experiment = experiment
            self.events = experiment.events

        self.file_date = filename.replace("events/", "").replace("experiment", "").replace(".json", "").replace(".gz", "")

        for event in self.events:
            try:
                self.processes[event.pid]
            except KeyError:
                self.processes[event.pid] = []
            self.processes[event.pid].append(event)

        for pid in self.processes:
            for i in range(len(self.processes[pid]) - 1):
                event_a = self.processes[pid][i]
                event_b = self.processes[pid][i + 1]
                type_name = event_a.probe_point + ":" + event_b.probe_point
                try:
                    self.intervals[type_name]
                except KeyError:
                    self.intervals[type_name] = []
                self.intervals[type_name].append(
                    Interval(event_b.timestamp - event_a.timestamp, event_a, event_b, pid, event_a.tgid))

        for interval_type in self.intervals.keys():
            if interval_type not in self.interval_types:
                self.interval_types.append(interval_type)

        for event in self.events:
            try:
                self.event_count[event.probe_point]
            except KeyError:
                self.event_count[event.probe_point] = 0
            self.event_count[event.probe_point] += 1

    def __fill_events_per_process(self):
        if self.events_per_process:
            # was already filled
            return
        for pid in self.processes:
            num_events = len([x for x in self.events if x.pid == pid])
            # print(f"In {pid} are {num_events} events.")#
            self.events_per_process[pid] = num_events

    def check_events_per_process(self) -> bool:
        self.__fill_events_per_process()
        mean = np.mean(list(self.events_per_process.values()))
        standard_deviation = np.std(list(self.events_per_process.values()))

        problem = False
        for pid, events in self.events_per_process.items():
            if np.abs(events - mean) > (5 * standard_deviation):
                if not problem:
                    print(f"Arithmetic mean is {mean}.")
                    print(f"Standard deviation is {standard_deviation}.")
                print(f"PID {pid}'s # of events ({events}) differs from the mean ({mean}) more than 5x the standard deviation.")
                problem = True
        return problem

    def sanity_check(self):
        problem = self.check_events_per_process()  # or self.check*
        if problem:
            print("#############################################################")
            print("THIS DATASET DID NOT PASS THE SANITY CHECK! IT MAY MISS DATA!")
            print("#############################################################")

parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", default="events", type=str, help="Directory containing event data.")

args = parser.parse_args()

if not os.path.isdir(args.directory):
    print("Error: " + args.directory + " is not a valid directory.")
    exit()

# Iterate through each file in the directory
for filename in os.listdir(args.directory):
    filepath = os.path.join(args.directory, filename)
    
    # Check if it's a file (not a directory or symbolic link)
    if os.path.isfile(filepath):
        iv = Intervals(filepath)
        iv.sanity_check()

print("Processed all files from " + args.directory)
