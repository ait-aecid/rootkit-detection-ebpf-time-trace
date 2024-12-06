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
#import pandasql as psql
from data_classes import Event, Interval, experiment_from_json
from scipy.stats import ttest_ind
from scipy.stats import wasserstein_distance_nd
from scipy.stats import norm

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

        # To decrease memory usage of each object, remove events since they are not needed for anomaly detection
        # Comment out the following two lines to analyze event data rather than interval data
        self.events = []
        self.experiment.events = []

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

#def shift_detection(train_quantiles_dict, test, quantiles):
#    for name in set(train_quantiles_dict.keys()).intersection(set(test.keys())):
#        # Only check names that appear both in train and test
#        train_quantiles = train_quantiles_dict[name]
#        test_quantiles = np.quantile(test[name], quantiles)
#        #diff = np.subtract(test_quantiles, train_quantiles)
#        diff = np.subtract(test_quantiles, train_quantiles) # / np.median(train_quantiles) #np.divide(np.subtract(test_quantiles, train_quantiles), train_quantiles)
#        if name == "filldir64-return:filldir64-enter":
#        #    #print(train_quantiles)
#        #    #print(test_quantiles)
#            print(diff)
#            return diff

def get_fone(tp, fn, tn, fp):
    # Compute the F1 score based on detected samples
    if tp + fp + fn == 0:
        return 'inf'
    return tp / (tp + 0.5 * (fp + fn))

def print_results(name, tp, fn, tn, fp, threshold, det_time):
    # Compute metrics and return a dictionary with results
    if tp + fn == 0:
        tpr = "inf"
    else:
        tpr = tp / (tp + fn)
    if fp + tn == 0:
        fpr = "inf"
    else:
        fpr = fp / (fp + tn)
    if tn + fp == 0:
        tnr = "inf"
    else:
        tnr = tn / (tn + fp)
    if tp + fp == 0:
        p = "inf"
    else:
        p = tp / (tp + fp)
    fone = get_fone(tp, fn, tn, fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = "inf"
    if tp + fp != 0 and tp + fn != 0 and tn + fp != 0 and tn + fn != 0:
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    print('')
    print(name)
    if threshold is not None:
        print(' Threshold=' + str(threshold))
    else:
        threshold = -1
    print(' Time=' + str(det_time))
    print(' TP=' + str(tp))
    print(' FP=' + str(fp))
    print(' TN=' + str(tn))
    print(' FN=' + str(fn))
    print(' TPR=R=' + str(tpr))
    print(' FPR=' + str(fpr))
    print(' TNR=' + str(tnr))
    print(' P=' + str(p))
    print(' F1=' + str(fone))
    print(' ACC=' + str(acc))
    print(' MCC=' + str(mcc))
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'p': p, 'f1': fone, 'acc': acc, 'threshold': threshold, 'name': name, 'time': det_time}

def get_diffs(batches_a, batches_b, quantiles):
    diffs = {}
    for batch_a in batches_a:
        for name, interv_list_a in batch_a.intervals_time.items():
            if name not in diffs:
                diffs[name] = {}
                for q in quantiles:
                    diffs[name][q] = []
            interv_list_quantiles_a = np.quantile(interv_list_a, quantiles)
            for batch_b in batches_b:
                if batch_a == batch_b:
                    continue
                if name not in batch_b.intervals_time:
                    continue
                interv_list_b = batch_b.intervals_time[name]
                interv_list_quantiles_b = np.quantile(interv_list_b, quantiles)
                diff = np.subtract(interv_list_quantiles_a, interv_list_quantiles_b)
                for i, q in enumerate(quantiles):
                    diffs[name][q].append(diff[i])
    return diffs

def get_quantile_vals(batches, quantiles):
    vals = {}
    for batch in batches:
        for name, interv_list in batch.intervals_time.items():
            if name not in vals:
                vals[name] = {}
                for q in quantiles:
                    vals[name][q] = []
            interv_list_quantiles = np.quantile(interv_list, quantiles)
            for i, q in enumerate(quantiles):
                vals[name][q].append(interv_list_quantiles[i])
    return vals

def get_crits(vals_a, vals_b, quantiles):
    crits = {}
    anom = False
    for name in set(vals_a.keys()).intersection(set(vals_b.keys())):
        crits[name] = {}
        for q in quantiles:
            #if len(vals_a[name][q]) < 10 or len(vals_b[name][q]) < 10:
            #    continue
            #t_stat, p_val = ttest_ind(vals_a[name][q], vals_b[name][q])
            #print(len(vals_b[name][q]), len(vals_a[name][q]))
            if len(vals_a[name][q]) < 5:
                continue
            # cdf has mean at 0.5 and min/max at 0/1; compute score that is 1 at the mean and approaches 0 at the tails
            crit = 1 - 2 * np.abs(norm.cdf(vals_b[name][q], loc=np.mean(vals_a[name][q]), scale=np.std(vals_a[name][q])) - 0.5)
            crits[name][q] = crit
            #if crit <= 1: #0.001: #1e-20:
            ##    anom = True
            #    print(name, q, crit)
            #    print(" " + str(vals_a[name][q]))
            #    print(" " + str(vals_b[name][q]))
    return crits #, anom

def run_test(train, test_norm, test_anom, quantiles):
    # Generate model by computing shifts between pairs of batches
    #train_diffs = get_diffs(train, train, quantiles)
    train_vals = get_quantile_vals(train, quantiles)
    #for train_batch in train:
    #    for name, interv_list in train_batch.intervals_time.items():
    #        interv_list_quantiles = np.quantile(interv_list, quantiles)
    #        for train_batch_inner in train:
    #            if train_batch == train_batch_inner:
    #                continue
    #            interv_list_inner = train_batch_inner.intervals_time[name]
    #            interv_list_inner_quantiles = np.quantile(interv_list_inner, quantiles)
    #            diff = np.subtract(interv_list_quantiles, interv_list_inner_quantiles)
    #            if name in train_merge:
    #                train_merge[name].extend(diff)
    #            else:
    #                train_merge[name] = diff
    #print(train_diffs)

    #tp, fp, tn, fn = 0, 0, 0, 0
    # Test normal data
    #for test_norm_batch in test_norm:
    #    #test_norm_diffs = get_diffs(train, [test_norm_batch], quantiles)
    #    test_norm_vals = get_quantile_vals([test_norm_batch], quantiles)
    #    p_vals, anom = get_p_vals(train_vals, test_norm_vals, quantiles)
    #    if anom:
    #        fp += 1
    #    else:
    #        tn += 1
    #for name in set(train_diffs.keys()).intersection(set(test_norm_diffs.keys())):
    #    for q in quantiles:
    #        t_stat, p_val = ttest_ind(train_diffs[name][q], test_norm_diffs[name][q])

    #for test_norm_batch in test_norm:
    #    for name, interv_list in test_norm_batch.intervals_time.items():
    #        test_norm_quantiles = np.quantile(interv_list, quantiles)
    #        train_diffs = train_merge[name]
    #    x = shift_detection(train_quantiles, test_norm_batch.intervals_time, quantiles)
    #    print(test_norm_batch.experiment.label + ': ' + str(np.median(x)) + ", " + str(np.std(x)))

    # Test anomalous data
    #exit()
    #print("\nanom\n")
    #for test_anom_batch in test_anom:
    #    #test_anom_diffs = get_diffs(train, [test_anom_batch], quantiles)
    #    test_anom_vals = get_quantile_vals([test_anom_batch], quantiles)
    #    p_vals, anom = get_p_vals(train_vals, test_anom_vals, quantiles)
    #    if anom:
    #        tp += 1
    #    else:
    #        fn += 1
    #for test_anom_batch in test_anom:
    #    x = shift_detection(train_quantiles, test_anom_batch.intervals_time, quantiles)
    #    print(test_anom_batch.experiment.label + ': ' + str(np.median(x)) + ", " + str(np.std(x)))
    crits_norm, crits_anom = [], []
    for test_norm_batch in test_norm:
        test_norm_vals = get_quantile_vals([test_norm_batch], quantiles)
        crits_norm.append(get_crits(train_vals, test_norm_vals, quantiles))
        #exit()
    #print(crits_norm)
    for test_anom_batch in test_anom:
        test_anom_vals = get_quantile_vals([test_anom_batch], quantiles)
        crits_anom.append(get_crits(train_vals, test_anom_vals, quantiles))
    best_metrics = {"fone": None, "tp": None, "fp": None, "tn": None, "fn": None, "time": None, "thresh": None}
    for thresh in np.logspace(-30, 0, num=100):
        start_time = time.time()
        tp, fp, tn, fn = 0, 0, 0, 0
        for batch_crits in crits_norm:
            anom = False
            for name, quantile_dict in batch_crits.items():
                for q, crit in quantile_dict.items():
                    if crit < thresh:
                        #print(crit)
                        anom = True
                        break
                if anom:
                    break
            if anom:
                fp += 1
            else:
                tn += 1
        for batch_crits in crits_anom:
            anom = False
            for name, quantile_dict in batch_crits.items():
                for q, crit in quantile_dict.items():
                    if crit < thresh:
                        #print(crit)
                        anom = True
                        break
                if anom:
                    break
            if anom:
                tp += 1
            else:
                fn += 1
        fone = get_fone(tp, fn, tn, fp)
        #print(thresh, fone)
        total_time = time.time() - start_time
        if best_metrics["fone"] is None or fone >= best_metrics["fone"]:
            best_metrics["fone"] = fone
            best_metrics["tp"] = tp
            best_metrics["fp"] = fp
            best_metrics["tn"] = tn
            best_metrics["fn"] = fn
            best_metrics["time"] = total_time
            best_metrics["thresh"] = thresh
    print_results("test", best_metrics["tp"], best_metrics["fn"], best_metrics["tn"], best_metrics["fp"], best_metrics["thresh"], best_metrics["time"])

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

quantiles = np.linspace(0, 1 - 1 / args.quantiles, args.quantiles) # Excludes 1 to avoid last term (which is usually an outlier), e.g., for args.quantiles = 100 will result in 0, 0.01, 0.02, ..., 0.99
run_test(ivs_train, ivs_test_norm, ivs_test_anom, quantiles)
