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

class Intervals:
    def __init__(self, filename):
        self.args = []
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

        experiment = None
        with gzip.open(filename, 'r') as file:
            json_obj = json.load(file)
            experiment = experiment_from_json(json_obj)

        self.file_date = filename.replace("events/", "").replace("experiment", "").replace(".json", "").replace(".gz", "")

        processes = {}
        for event in experiment.events:
            try:
                processes[event.pid]
            except KeyError:
                processes[event.pid] = []
            processes[event.pid].append(event)

        for pid in processes:
            for i in range(len(processes[pid]) - 1):
                event_a = processes[pid][i]
                event_b = processes[pid][i + 1]
                type_name = event_a.probe_point + ":" + event_b.probe_point
                try:
                    self.intervals_time[type_name]
                except KeyError:
                    self.intervals_time[type_name] = []
                self.intervals_time[type_name].append(event_b.timestamp - event_a.timestamp)

        for event in experiment.events:
            try:
                self.event_count[event.probe_point]
            except KeyError:
                self.event_count[event.probe_point] = 0
            self.event_count[event.probe_point] += 1
        
        # Remove events from experiment to avoid that python runs out of RAM
        experiment.events = []
        self.experiment = experiment

    def __fill_events_per_process(self):
        if self.events_per_process:
            # was already filled
            return
        for pid in self.processes:
            num_events = len([x for x in self.events if x.pid == pid])
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
        for name, interv in self.intervals_time.items():
            for cnt, delta in enumerate(interv):
                #out.write(str(self.filename) + ',' + str(name) + ',' + str(cnt) + ',' + str(i.time) + ',' + str(i.pid) + ',' + str(i.tgid) + ',' + str(self.experiment.label) + ',' + str(self.experiment.description) + '\n')
                out.write(str(self.filename) + ',' + str(name) + ',' + str(cnt) + ',' + str(delta) + ',' + str(self.experiment.label) + ',' + str(self.experiment.description) + '\n')

def export_all_intervals_to_csv(ivs):
    with open('intervals.csv', 'w+') as out:
        #out.write('filename,name,id,delta,pid,tgit,label,description\n')
        out.write('filename,name,id,delta,label,description\n')
        for label, ivs_dict in ivs.items():
            for description, iv_list in ivs_dict.items():
                for iv in iv_list:
                    iv.export_intervals_to_csv(out)

def get_fone(tp, fn, tn, fp):
    # Compute the F1 score based on detected samples
    if tp + fp + fn == 0:
        return 'inf'
    return tp / (tp + 0.5 * (fp + fn))

def compute_results(do_print, name, tp, fn, tn, fp, threshold, det_time):
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
    if do_print:
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
        print("")
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'tpr': tpr, 'fpr': fpr, 'tnr': tnr, 'p': p, 'f1': fone, 'acc': acc, 'threshold': threshold, 'name': name, 'time': det_time}

def print_confusion(name, tp_c, fn_c, tn_c, fp_c, run, out_c):
    print(name)
    print("Predicted")
    s = ""
    for description_train, inner_dict in tp_c.items():
        s += description_train + "\t"
        if len(description_train) < 8:
            # Extra tab for short descriptions
            s += "\t"
    print(s + "\n" + str("Pos\tNeg\t" * len(tp_c)))
    for description_train, inner_dict in tp_c.items():
        s = ""
        for description in inner_dict:
            s += str(tp_c[description_train][description]) + "\t" + str(fn_c[description_train][description]) + "\t"
            out_c.write(str(run) + "," + description_train + ",Pos" + "," + description + ",Pos" + "," + str(tp_c[description_train][description]) + "\n")
            out_c.write(str(run) + "," + description_train + ",Neg" + "," + description + ",Pos" + "," + str(fn_c[description_train][description]) + "\n")
            out_c.write(str(run) + "," + description_train + ",Pos" + "," + description + ",Neg" + "," + str(fp_c[description_train][description]) + "\n")
            out_c.write(str(run) + "," + description_train + ",Neg" + "," + description + ",Neg" + "," + str(tn_c[description_train][description]) + "\n")
            #out_c.write(str(run) + "," + description_train + "," + description + "," + str(tp_c[description_train][description]) + "," + str(fp_c[description_train][description]) + "," + str(tn_c[description_train][description]) + "," + str(fn_c[description_train][description]) + "\n")
        s += "\tPos - Actual\t" + description_train + "\n"
        for description in inner_dict:
            s += str(fp_c[description_train][description]) + "\t" + str(tn_c[description_train][description]) + "\t"
        print(s + "\tNeg - Actual\t" + description_train)
    print("")

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
    num_obs = {}
    for batch in batches:
        for name, interv_list in batch.intervals_time.items():
            if name not in vals:
                vals[name] = [np.quantile(interv_list, quantiles)]
                num_obs[name] = len(interv_list)
            else:
                vals[name].append(np.quantile(interv_list, quantiles))
                num_obs[name] += len(interv_list)
    return vals, num_obs

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
        test_vals_scaled = (test_vals[name] - train_mean[name]) / np.sqrt(train_var[name])
        mhd = np.sum(test_vals_scaled**2)
        p_value = 1 - chi2.cdf(mhd, df=len(quantiles))
        #print(name, p_value, np.mean(test_vals[name] - train_mean[name]))
        crits[name] = p_value
    return crits

def get_stats(data, num_batches):
    mean = {}
    var = {}
    cov = {}
    cov_inv = {}
    for name, vals in data.items():
        #if len(vals) < num_batches / 2:
        #    # Name only appears in some of the batches; likely a result of incorrectly collected intervals 
        #    continue
        mean[name] = np.mean(vals, axis=0)
        var[name] = np.var(vals, axis=0, ddof=0)
    return mean, var

def run_test(train, test, quantiles, run, out_best, out_all, out_c):
    train_num_obs = {}
    train_mean = {}
    train_var = {}
    #descriptions = set()
    for description, train_batches in train.items():
        #descriptions.add(description)
        train_vals, train_num_obs[description] = get_quantile_vals(train_batches, quantiles)
        train_mean[description], train_var[description] = get_stats(train_vals, len(train_batches))
    crits = {}
    for label in test: #["normal"]: #test:
        for description_train in train_mean: # ["default"]: # train_mean:
            # Interate through all training models
            for description, test_batches in test[label].items():
                #if description != "filename_length":
                #    continue
                # For each training model, iterate through all test values
                if label not in crits:
                    crits[label] = {}
                if description_train not in crits[label]:
                    crits[label][description_train] = {}
                if description not in crits[label][description_train]:
                    crits[label][description_train][description] = []
                for i, test_batch in enumerate(test_batches):
                    print(i)
                    test_vals, test_num_obs = get_quantile_vals([test_batch], quantiles)
                    crits[label][description_train][description].append(get_crits(train_mean[description_train], train_var[description_train], train_num_obs[description_train], test_vals, test_num_obs, quantiles))
    best_metrics = {"fone": None, "tp": None, "fp": None, "tn": None, "fn": None, "time": None, "thresh": None, "name_counts": None}
    for thresh in np.logspace(-30, 0, num=100):
        start_time = time.time()
        tp, fp, tn, fn = 0, 0, 0, 0 # Counts differentiate only normal and anomalous classes, independent from sub-classes
        tp_c, fp_c, tn_c, fn_c = {}, {}, {}, {} # Use sub-classes for the confusion matrix
        name_counts = {} # Counts which function pairs are the ones most often reporting anomalies
        #for description in descriptions:
        #    tp_c[description] = 0
        #    fp_c[description] = 0
        #    tn_c[description] = 0
        #    fn_c[description] = 0
        for label in crits:
            if label not in name_counts:
                name_counts[label] = {}
            for description_train, crits_inner in crits[label].items():
                if description_train not in tp_c:
                    tp_c[description_train] = {}
                    fp_c[description_train] = {}
                    tn_c[description_train] = {}
                    fn_c[description_train] = {}
                for description, crit_list in crits_inner.items():
                    if description not in tp_c[description_train]:
                        tp_c[description_train][description] = 0
                        fp_c[description_train][description] = 0
                        tn_c[description_train][description] = 0
                        fn_c[description_train][description] = 0
                    for crit_dict in crit_list:
                        anomaly_detected = False
                        for name, crit in crit_dict.items():
                            if crit < thresh:
                                anomaly_detected = True
                                if name not in name_counts[label]:
                                    name_counts[label][name] = 0
                                name_counts[label][name] += 1
                                #break
                        if anomaly_detected:
                            # Detected as anomaly
                            if label == rootkit_key:
                                if description_train == description:
                                    tp += 1
                                tp_c[description_train][description] += 1
                            else:
                                if description_train == description:
                                    fp += 1
                                fp_c[description_train][description] += 1
                        else:
                            # Detected as normal
                            if label == rootkit_key:
                                if description_train == description:
                                    fn += 1
                                fn_c[description_train][description] += 1
                            else:
                                if description_train == description:
                                    tn += 1
                                tn_c[description_train][description] += 1
        fone = get_fone(tp, fn, tn, fp)
        res_tmp = compute_results(False, "not_print", tp, fn, tn, fp, thresh, -1)
        out_all.write(str(run) + "," + str(fone) + "," + str(tp) + "," + str(fp) + "," + str(tn) + "," + str(fn) + "," + str(-1) + "," + str(len(quantiles)) + "," + str(thresh) + "," + str(res_tmp["tpr"]) + "," + str(res_tmp["fpr"]) + "," + str(res_tmp["tnr"]) + "," + str(res_tmp["p"]) + "," + str(res_tmp["acc"]) + "\n")
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
            best_metrics["name_counts"] = name_counts
    res = compute_results(True, "Results (Run " + str(run) + ")", best_metrics["tp"], best_metrics["fn"], best_metrics["tn"], best_metrics["fp"], best_metrics["thresh"], best_metrics["time"])
    print_confusion("Confusion Matrix (Run " + str(run) + ")", best_metrics["tp_c"], best_metrics["fn_c"], best_metrics["tn_c"], best_metrics["fp_c"], run, out_c)
    out_best.write(str(run) + "," + str(res["f1"]) + "," + str(res["tp"]) + "," + str(res["fp"]) + "," + str(res["tn"]) + "," + str(res["fn"]) + "," + str(res["time"]) + "," + str(len(quantiles)) + "," + str(res["threshold"]) + "," + str(res["tpr"]) + "," + str(res["fpr"]) + "," + str(res["tnr"]) + "," + str(res["p"]) + "," + str(res["acc"]) + "\n")
    if False:
        print("Function pairs that reported most anomalies:")
        for label, name_counts_dict in best_metrics["name_counts"].items():
            print(label)
            for name, cnt in name_counts_dict.items():
                print("  " + name + ": " + str(cnt))
    return best_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", default="events", type=str, help="Directory containing event data.")
parser.add_argument("--train_ratio", "-t", default=0.333, help="Fraction of normal data used for training.", type=float)
parser.add_argument("--seed", "-s", default=None, help="Seed for random sampling.", type=int)
parser.add_argument("--quantiles", "-q", default=10, help="Number of quantiles.", type=int)
parser.add_argument("--repeat", "-r", default=1, help="Repeat experiment with different training samples multiple times (only in offline mode).", type=int)
parser.add_argument("--mode", "-m", default="offline", choices=["offline", "online"], help="Evaluate mode.", type=str)

args = parser.parse_args()

random.seed(args.seed)

if not os.path.isdir(args.directory):
    print("Error: " + args.directory + " is not a valid directory.")
    exit()

ivs = {normal_key: {}, rootkit_key: {}}
files = os.listdir(args.directory)
files.sort()
for filename in tqdm(files):
    filepath = os.path.join(args.directory, filename)
    
    # Check if it's a file (not a directory or symbolic link)
    if os.path.isfile(filepath):
        iv = Intervals(filepath)
        #iv.sanity_check()
        if iv.experiment.label == normal_key:
            if iv.experiment.description not in ivs[normal_key]:
                ivs[normal_key][iv.experiment.description] = []
            ivs[normal_key][iv.experiment.description].append(iv)
        elif iv.experiment.label == rootkit_key:
            if iv.experiment.description not in ivs[rootkit_key]:
                ivs[rootkit_key][iv.experiment.description] = []
            ivs[rootkit_key][iv.experiment.description].append(iv)

print("Processed all files from " + args.directory)
print("")

#export_all_intervals_to_csv(ivs)

if args.mode == "offline":
    with open("results_offline_best.csv", "w+") as out_best, open("results_offline_all.csv", "w+") as out_all, open("results_offline_confusion.csv", "w+") as out_c:
        out_best.write("run,fone,tp,fp,tn,fn,time,q,thresh,tpr,fpr,tnr,p,acc\n")
        out_all.write("run,fone,tp,fp,tn,fn,time,q,thresh,tpr,fpr,tnr,p,acc\n")
        #out_c_str = ""
        #for description in ivs[normal_key]:
        #    out_c_str += description + "_tp," + description + "_fp," + description + "_tn," + description + "_fn,"
        #out_c.write(out_c_str[:-1] + "\n")
        out_c.write("run,pred,pred_class,actual,actual_class,cnt\n")
        for run in range(args.repeat):
            run += 1 # Start with run #1
            # Get training data from normal data (default case) and remove it from test data
            ivs_train = {}
            for description in ivs[normal_key]:
                random.shuffle(ivs[normal_key][description])
                split_point = math.ceil(len(ivs[normal_key][description]) * args.train_ratio)
                ivs_train[description] = ivs[normal_key][description][:split_point]
                ivs[normal_key][description] = ivs[normal_key][description][split_point:]
            
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
            
            if args.quantiles > 0:
                quantiles = np.linspace(0, 1 - 1 / args.quantiles, args.quantiles) # Excludes 1 to avoid last term (which is usually an outlier), e.g., for args.quantiles = 100 will result in 0, 0.01, 0.02, ..., 0.99
            else:
                # This case is just used to test teh influence of the number of quantiles
                quantiles = np.linspace(0, 1 - 1 / run, run) # Increase the number of quantiles by 1 in every run
            best_metrics = run_test(ivs_train, ivs, quantiles, run, out_best, out_all, out_c)
        
            # Return training data to normal data in case that there is another iteration
            for description in ivs_train:
                ivs[normal_key][description].extend(ivs_train[description])


