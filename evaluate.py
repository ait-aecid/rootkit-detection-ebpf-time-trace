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
from scipy.stats import chi2
from tqdm import tqdm

normal_key = "normal"
rootkit_key = "rootkit"

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
                out.write(str(self.filename) + ',' + str(name) + ',' + str(cnt) + ',' + str(i.time) + ',' + str(i.pid) + ',' + str(i.tgid) + ',' + str(self.experiment.label) + ',' + str(self.experiment.description) + '\n')

    def export_event_counts(self, out):
        counts = {}
        for event in self.experiment.events:
            if event.probe_point not in counts:
                counts[event.probe_point] = 1
            else:
                counts[event.probe_point] += 1
        for probe_point_name, cnt in counts.items():
            out.write(str(self.filename) + ',' + str(probe_point_name) + ',' + str(cnt) + ',' + str(self.experiment.label) + '\n')

def export_all_intervals_to_csv(ivs):
    with open('intervals.csv', 'w+') as out:
        out.write('filename,name,id,delta,pid,tgit,label,description\n')
        for label, ivs_dict in ivs.items():
            for description, iv_list in ivs_dict.items():
                for iv in iv_list:
                    iv.export_intervals_to_csv(out)

def export_all_event_counts(ivs):
    with open('event_counts.csv', 'w+') as out:
        out.write('filename,event,count,label\n')
        for label, ivs_dict in ivs.items():
            for description, iv_list in ivs_dict.items():
                for iv in iv_list:
                    iv.export_event_counts(out)

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

def print_confusion(name, tp_c, fn_c, tn_c, fp_c):
    print(name)
    print("Predicted")
    print("Pos\tNeg")
    for description in tp_c:
        print(str(tp_c[description]) + "\t" + str(fn_c[description]) + "\tPos - Actual " + description)
        print(str(fp_c[description]) + "\t" + str(tn_c[description]) + "\tNeg - Actual " + description)

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

def get_quantile_vals_old(batches, quantiles):
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

def get_quantile_vals(batches, quantiles):
    vals = {}
    num_obs = {}
    for batch in batches:
        for name, interv_list in batch.intervals_time.items():
            if name not in vals:
                vals[name] = [np.quantile(interv_list, quantiles)]
                num_obs[name] = len(interv_list)
            else:
                vals[name].append(np.quantile(interv_list, quantiles))
                num_obs[name] += len(interv_list)
                #for q in quantiles:
                #    vals[name][q] = []
            #interv_list_quantiles = np.quantile(interv_list, quantiles)
            #for i, q in enumerate(quantiles):
            #    vals[name][q].append(interv_list_quantiles[i])
    return vals, num_obs

def get_crits_old(vals_a, vals_b, quantiles):
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
            if crit <= 0.001: #1e-20:
            #    anom = True
                print(name, q, crit)
                print(" " + str(vals_a[name][q]))
                print(" " + str(vals_b[name][q]))
    return crits #, anom

#def get_crits(train_mean, train_var, train_cov, train_cov_inv, train_num_obs, test_vals, test_num_obs, quantiiles):
def get_crits(train_mean, train_var, train_num_obs, test_vals, test_num_obs, quantiiles):
    crits = {}
    for name in train_mean:
        if name not in test_vals:
            continue
        if train_num_obs[name] < len(quantiles):
            continue
        if test_num_obs[name] < len(quantiles):
            continue
        crits[name] = {}
        ##diff = test_vals[name] - train_mean[name]
        #print(test_vals[name])
        #print("x")
        #print(train_mean[name])
        #print("y")
        #print(diff)
        #asdf()
        #train_cov_inv = np.linalg.inv(train_cov[name])
        #print("")
        #print(name)
        #print(diff.T)
        #print(train_cov_inv[name])
        #print(diff)
        #print(train_cov_inv[name].shape)
        ##mhd = diff @ train_cov_inv[name] @ diff.T

        test_vals_scaled = (test_vals[name] - train_mean[name]) / np.sqrt(train_var[name])
        mhd = np.sum(test_vals_scaled**2)
        # Compute the p-value
        p_value = 1 - chi2.cdf(mhd, df=len(quantiles))
        #if p_value < 1e-20:
        #    print(name, p_value, train_num_obs[name], test_num_obs[name], train_mean[name], np.sqrt(train_var[name]), test_vals[name])
        print(name, p_value, train_num_obs[name], test_num_obs[name])
        #print(name, train_cov_inv[name].shape[0], np.linalg.cond(train_cov[name]), np.linalg.eigvals(train_cov[name]), mhd, p_value)
        #print(name, len(quantiles), train_cov_inv[name].shape[0], np.linalg.cond(train_cov[name]), train_num_obs[name], test_num_obs[name], mhd, p_value)
        crits[name] = p_value
        #for q in quantiles:
        #    #if len(vals_a[name][q]) < 10 or len(vals_b[name][q]) < 10:
        #    #    continue
        #    #t_stat, p_val = ttest_ind(vals_a[name][q], vals_b[name][q])
        #    #print(len(vals_b[name][q]), len(vals_a[name][q]))
        #    if len(vals_a[name][q]) < 5:
        #        continue
        #    # cdf has mean at 0.5 and min/max at 0/1; compute score that is 1 at the mean and approaches 0 at the tails
        #    crit = 1 - 2 * np.abs(norm.cdf(vals_b[name][q], loc=np.mean(vals_a[name][q]), scale=np.std(vals_a[name][q])) - 0.5)
        #    crits[name][q] = crit
        #    if crit <= 0.001: #1e-20:
        #    #    anom = True
        #        print(name, q, crit)
        #        print(" " + str(vals_a[name][q]))
        #        print(" " + str(vals_b[name][q]))
    return crits #, anom

def get_stats(data, num_batches):
    mean = {}
    var = {}
    cov = {}
    cov_inv = {}
    for name, vals in data.items():
        #print(name, len(vals))
        if len(vals) < num_batches / 2:
            # Name only appears in some of the batches; likely a result of incorrectly collected intervals 
            continue
        #print(name)
        #print(vals)
        #print(np.array(vals).shape, len(vals))
        mean[name] = np.mean(vals, axis=0)
        var[name] = np.var(vals, axis=0, ddof=0)
        #print(mean[name])
        #zero_mean_data = vals - mean[name]
        #print(zero_mean_data)
        #cov[name] = np.cov(zero_mean_data, rowvar=False)
        #if True:
        #    cov[name] = cov[name] + 1e-6 * np.eye(cov[name].shape[0])
        #print(cov)
        #asdf()
        #print(cov[name])
        #cov_inv[name] = np.linalg.inv(cov[name])
    return mean, var #, cov, cov_inv

def run_test(train, test, quantiles):
    # Generate model by computing shifts between pairs of batches
    #train_diffs = get_diffs(train, train, quantiles)
    train_num_obs = {}
    train_mean = {}
    train_var = {}
    for description, train_batches in train.items():
        #print(description)
        train_vals, train_num_obs[description] = get_quantile_vals(train_batches, quantiles)
        #train_mean, train_var, train_cov, train_cov_inv = get_stats(train_vals)
        train_mean[description], train_var[description] = get_stats(train_vals, len(train_batches))
    #asdf()
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
    crits = {}
    descriptions = set()
    i = 0
    for label in test: #["normal"]: #test:
        for description, test_batches in test[label].items():
            #test_norm_vals, test_norm_num_obs = get_quantile_vals([test_norm_batch], quantiles)
            if label not in crits:
                crits[label] = {}
            if description not in crits[label]:
                crits[label][description] = []
            descriptions.add(description)
            for test_batch in test_batches:
                print(i)
                i += 1
                test_vals, test_num_obs = get_quantile_vals([test_batch], quantiles)
                #crits[label][description].append(get_crits(train_mean, train_var, train_cov, train_cov_inv, train_num_obs, test_vals, test_num_obs, quantiles))
                crits[label][description].append(get_crits(train_mean[description], train_var[description], train_num_obs[description], test_vals, test_num_obs, quantiles))
    #crits_norm, crits_anom = [], []
    #for test_norm_batch in test_norm:
    #    test_norm_vals, test_norm_num_obs = get_quantile_vals([test_norm_batch], quantiles)
    #    crits_norm.append(get_crits(train_mean, train_var, train_cov, train_cov_inv, train_num_obs, test_norm_vals, test_norm_num_obs, quantiles))
    #    #print("")
    #    #exit()
    ##print("\nANOM\n")
    ##print(crits_norm)
    #for test_anom_batch in test_anom:
    #    test_anom_vals, test_anom_num_obs = get_quantile_vals([test_anom_batch], quantiles)
    #    crits_anom.append(get_crits(train_mean, train_var, train_cov, train_cov_inv, train_num_obs, test_anom_vals, test_anom_num_obs, quantiles))
    #    #print("")
    #    #asdf()
    #sub_classes = set()
    #for batch in train + test_norm + test_anom:
    #    sub_classes.add(batch.experiment.description)
    best_metrics = {"fone": None, "tp": None, "fp": None, "tn": None, "fn": None, "time": None, "thresh": None}
    for thresh in np.logspace(-100, 0, num=100):
        start_time = time.time()
        tp, fp, tn, fn = 0, 0, 0, 0 # Counts differentiate only normal and anomalous classes, independent from sub-classes
        tp_c, fp_c, tn_c, fn_c = {}, {}, {}, {} # Use sub-classes for the confusion matrix
        for description in descriptions:
            tp_c[description] = 0
            fp_c[description] = 0
            tn_c[description] = 0
            fn_c[description] = 0
        for label in crits:
            for description, crit_list in crits[label].items():
                for crit_dict in crit_list:
                    anomaly_detected = False
                    for name, crit in crit_dict.items():
                        if crit < thresh:
                            anomaly_detected = True
                            break
                    if anomaly_detected:
                        # Detected as anomaly
                        if label == rootkit_key:
                            tp += 1
                            tp_c[description] += 1
                        else:
                            fp += 1
                            fp_c[description] += 1
                    else:
                        # Detected as normal
                        if label == rootkit_key:
                            fn += 1
                            fn_c[description] += 1
                        else:
                            tn += 1
                            tn_c[description] += 1
        #for batch_crits in crits_norm:
        #    anom = False
        #    for name, quantile_dict in batch_crits.items():
        #        for q, crit in quantile_dict.items():
        #            if crit < thresh:
        #                #print(crit)
        #                anom = True
        #                break
        #        if anom:
        #            break
        #    if anom:
        #        fp += 1
        #    else:
        #        tn += 1
        #for batch_crits in crits_anom:
        #    anom = False
        #    for name, quantile_dict in batch_crits.items():
        #        for q, crit in quantile_dict.items():
        #            if crit < thresh:
        #                #print(crit)
        #                anom = True
        #                break
        #        if anom:
        #            break
        #    if anom:
        #        tp += 1
        #    else:
        #        fn += 1
        fone = get_fone(tp, fn, tn, fp)
        #print(thresh, tp, fn, tn, fp, fone)
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
            best_metrics["tp_c"] = tp_c
            best_metrics["fn_c"] = fn_c
            best_metrics["tn_c"] = tn_c
            best_metrics["fp_c"] = fp_c
    print_results("test", best_metrics["tp"], best_metrics["fn"], best_metrics["tn"], best_metrics["fp"], best_metrics["thresh"], best_metrics["time"])
    print_confusion("test", best_metrics["tp_c"], best_metrics["fn_c"], best_metrics["tn_c"], best_metrics["fp_c"])

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

ivs = {normal_key: {}, rootkit_key: {}}
# Iterate through each file in the directory
files = os.listdir(args.directory)
#random.shuffle(files)
files.sort()
#files = files[:210] # TODO remove
for filename in tqdm(files):
    filepath = os.path.join(args.directory, filename)
    
    # Check if it's a file (not a directory or symbolic link)
    if os.path.isfile(filepath):
        iv = Intervals(filepath)
        #print(iv.experiment.label)
        #iv.sanity_check()
        #ivs.append(iv)
        if iv.experiment.label == normal_key:
            if iv.experiment.description not in ivs[normal_key]:
                ivs[normal_key][iv.experiment.description] = []
            ivs[normal_key][iv.experiment.description].append(iv)
        elif iv.experiment.label == rootkit_key:
            if iv.experiment.description not in ivs[rootkit_key]:
                ivs[rootkit_key][iv.experiment.description] = []
            ivs[rootkit_key][iv.experiment.description].append(iv)

print("Processed all files from " + args.directory)

#export_all_intervals_to_csv(ivs)
#export_all_event_counts(ivs)
#asdf()

#ivs_norm_default = []
#ivs_norm_other = []
#ivs_anom = []
#for iv in ivs:
#    if iv.experiment.label == "normal":
#        if iv.experiment.description == "default":
#            ivs_norm_default.append(iv)
#        else:
#            ivs_norm_other.append(iv)
#    else:
#        ivs_anom.append(iv)
#random.shuffle(ivs_norm_default)

# Get training data from normal data (default case) and remove it from test data
ivs_train = {}
for description in ivs[normal_key]:
    random.shuffle(ivs[normal_key][description])
    split_point = math.ceil(len(ivs[normal_key][description]) * args.train_ratio)
    ivs_train[description] = ivs[normal_key][description][:split_point]
    ivs[normal_key][description] = ivs[normal_key][description][split_point:]

#ivs_test_anom = ivs_anom

#ivs_norm = []
#ivs_anom = []
#for iv in ivs:
#    if iv.experiment.label == "normal":
#        ivs_norm.append(iv)
#    else:
#        ivs_anom.append(iv)
#random.shuffle(ivs_norm)
#split_point = math.ceil(len(ivs_norm) * args.train_ratio)
#ivs_train = ivs_norm[:split_point]
#ivs_test_norm = ivs_norm[split_point:]
#ivs_test_anom = ivs_anom

print("Normal batches: " + str(sum(len(value) for value in ivs[normal_key].values()) + sum(len(value) for value in ivs_train.values())))
print("  Normal batches for training: " + str(sum(len(value) for value in ivs_train.values())))
for description in ivs_train:
    print("    " + description + ": " + str(len(ivs_train[description])))
print("  Normal batches for testing: " + str(sum(len(value) for value in ivs[normal_key].values())))
for description in ivs[normal_key]:
    print("    " + description + ": " + str(len(ivs[normal_key][description])))
print("Anomalous batches: " + str(sum(len(value) for value in ivs[rootkit_key].values())))
for description in ivs[rootkit_key]:
    print("  " + description + ": " + str(len(ivs[rootkit_key][description])))

quantiles = np.linspace(0, 1 - 1 / args.quantiles, args.quantiles) # Excludes 1 to avoid last term (which is usually an outlier), e.g., for args.quantiles = 100 will result in 0, 0.01, 0.02, ..., 0.99
run_test(ivs_train, ivs, quantiles)
