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
import random
import numpy as np
import pandas as pd
import pandasql as psql
from data_classes import Event, Interval, experiment_from_json

class Intervals:
    def __init__(self, filename):
        self.args = []
        self.processes = {}
        self.intervals = {}
        self.intervals_time = {}
        self.interval_types = []
        self.file_date = ""
        self.experiment = None
        self.events = []
        self.events_per_process = {}
        self.dataframe = None
        self.event_count = {}
        self.filename = filename

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
                    self.intervals_time[type_name]
                except KeyError:
                    self.intervals[type_name] = []
                    self.intervals_time[type_name] = []
                self.intervals[type_name].append(
                    Interval(event_b.timestamp - event_a.timestamp, event_a, event_b, pid, event_a.tgid))
                self.intervals_time[type_name].append(event_b.timestamp - event_a.timestamp)

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

    def export_intervals_to_csv(self, out):
        for name, interv in self.intervals.items():
            for cnt, i in enumerate(interv):
                out.write(str(self.filename) + ',' + str(name) + ',' + str(cnt) + ',' + str(i.time) + ',' + str(i.pid) + ',' + str(i.tgid) + ',' + str(self.experiment.label) + '\n')

def export_all_intervals_to_csv(ivs):
    with open('intervals.csv', 'w+') as out:
        out.write('filename,name,id,delta,pid,tgit,label\n')
        for iv in ivs:
            iv.export_intervals_to_csv(out)

def shift_detection(train_quantiles_dict, test, quantiles):
    for name in set(train_quantiles_dict.keys()).intersection(set(test.keys())):
        # Only check names that appear both in train and test
        train_quantiles = train_quantiles_dict[name]
        test_quantiles = np.quantile(test[name], quantiles)
        #diff = np.subtract(test_quantiles, train_quantiles)
        diff = np.subtract(test_quantiles, train_quantiles) # / np.median(train_quantiles) #np.divide(np.subtract(test_quantiles, train_quantiles), train_quantiles)
        #if name == "filldir64-return:filldir64-enter":
        #    #print(train_quantiles)
        #    #print(test_quantiles)
        #    #print(diff)
        #    return diff


def run_test(train, test_norm, test_anom, quantiles):
    # Generate model by merging all training samples
    train_merge = {}
    for train_batch in train:
        for name, interv_list in train_batch.intervals_time.items():
            if name in train_merge:
                train_merge[name].extend(interv_list)
            else:
                train_merge[name] = interv_list
    train_quantiles = {}
    for name, interv_list in train_merge.items():
        train_quantiles[name] = np.quantile(interv_list, quantiles)

    # Test normal data
    for test_norm_batch in test_norm:
        x = shift_detection(train_quantiles, test_norm_batch.intervals_time, quantiles)
        print(test_norm_batch.experiment.label + ': ' + str(np.median(x)) + ", " + str(np.std(x)))

    # Test anomalous data
    for test_anom_batch in test_anom:
        x = shift_detection(train_quantiles, test_anom_batch.intervals_time, quantiles)
        print(test_anom_batch.experiment.label + ': ' + str(np.median(x)) + ", " + str(np.std(x)))

parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", default="events", type=str, help="Directory containing event data.")
parser.add_argument("--train_ratio", "-t", default=0.1, help="Fraction of normal data used for training.", type=float)
parser.add_argument("--seed", "-s", default=None, help="Seed for random sampling.", type=int)
parser.add_argument("--quantiles", "-q", default=100, help="Number of quantiles.", type=int)

args = parser.parse_args()

random.seed(args.seed)

if not os.path.isdir(args.directory):
    print("Error: " + args.directory + " is not a valid directory.")
    exit()

ivs = []
# Iterate through each file in the directory
for filename in os.listdir(args.directory):
    filepath = os.path.join(args.directory, filename)
    
    # Check if it's a file (not a directory or symbolic link)
    if os.path.isfile(filepath):
        iv = Intervals(filepath)
        print(iv.experiment.label)
        #iv.sanity_check()
        ivs.append(iv)

print("Processed all files from " + args.directory)

#export_all_intervals_to_csv(ivs)

ivs_norm = []
ivs_anom = []
for iv in ivs:
    if iv.experiment.label == "normal":
        ivs_norm.append(iv)
    else:
        ivs_anom.append(iv)
random.shuffle(ivs_norm)
split_point = math.ceil(len(ivs_norm) * args.train_ratio)
ivs_train = ivs_norm[:split_point]
ivs_test_norm = ivs_norm[split_point:]
ivs_test_anom = ivs_anom

print("Normal batches: " + str(len(ivs_norm)))
print("  Normal batches for training: " + str(len(ivs_train)))
print("  Normal batches for testing: " + str(len(ivs_test_norm)))
print("Anomalous batches: " + str(len(ivs_test_anom)))

quantiles = np.linspace(0, 1, args.quantiles)
run_test(ivs_train, ivs_test_norm, ivs_test_anom, quantiles)
