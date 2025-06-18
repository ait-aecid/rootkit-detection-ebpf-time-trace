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
from data_classes import Event, Interval, experiment_from_json
from scipy.stats import chi2
from tqdm import tqdm
from typing import *

normal_key = "normal"
rootkit_key = "rootkit"

parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", default="events", type=str, help="Directory containing event data.")
parser.add_argument("--train_ratio", "-t", default=0.333, help="Fraction of normal data used for training.", type=float)
parser.add_argument("--seed", "-s", default=None, help="Seed for random sampling.", type=int)
parser.add_argument("--quantiles", "-q", default=9, help="Number of quantiles.", type=int)
parser.add_argument("--repeat", "-r", default=1, help="Repeat experiment with different training samples multiple times (only in offline mode).", type=int)
parser.add_argument("--mode", "-m", default="offline", choices=["offline", "supervised", "online"], help="Evaluate mode.", type=str)
parser.add_argument("--grouping", "-g", default="fun", choices=["seq", "fun"], help="Grouping of events to interval either sequentially (independent of type and enter/return) or between enter and return of same function type.", type=str)
parser.add_argument("--approach", "-a", default="shift", choices=["shift", "ann", "lumped"], help="Approach used for detection - shift is based on Landauer et al. (2025), ann is based on Luckett et al. (2016), lumped is based on Lu et al. (2019).")
parser.add_argument("--export_intervals", "-e", action="store_true", help="Write intervals to file (change interval grouping mode with --grouping parameter)")

args = parser.parse_args()

if args.approach == "ann":
    print("Loading torch (required for --approach ann)...")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print("Loaded torch")

    class SampleEncoder(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
        def forward(self, x):
            return self.mlp(x)
    
    class DeepSetsEncoder(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=32, num_features=4):
            super().__init__()
            self.encoder = SampleEncoder(input_dim, hidden_dim)
            self.num_features = num_features
            self.output_dim = num_features * hidden_dim
    
        def forward(self, batch):
            batch_embeddings = []
            for features in batch:
                feature_embeddings = []
                for feature_set in features:
                    encoded = self.encoder(feature_set)
                    aggregated = encoded.mean(dim=0)
                    feature_embeddings.append(aggregated)
                batch_embedding = torch.cat(feature_embeddings, dim=0)
                batch_embeddings.append(batch_embedding)
            return torch.stack(batch_embeddings, dim=0)

random.seed(args.seed)

class Intervals:
    def __init__(self, filename, events_dir_name, grouping=None):
        self.args = []
        self.intervals = {}
        self.intervals_time = {}
        self.interval_types = []
        self.file_date = ""
        self.timestamp = ""
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

        self.file_date = filename.replace(events_dir_name + "/", "").replace("experiment", "").replace(".json", "").replace(".gz", "")
        self.timestamp = filename[(filename.find("T") + 1):filename.find(".")].replace("_", ":")

        processes = {}
        for event in experiment.events:
            try:
                processes[event.pid]
            except KeyError:
                processes[event.pid] = []
            processes[event.pid].append(event)
            
        if grouping == "seq":
            # Intervals are measured between neighboring events, independent from their type and enter/return
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
        elif grouping == "fun":
            # Intervals are measured between enter and return events of the same type
            # Note that events may be dropped when multiple enter events of the same type occur
            for pid in processes:
                start_events = {}
                for event in processes[pid]:
                    if event.probe_point.endswith("-enter"):
                        start_events[event.probe_point] = event.timestamp
                    elif event.probe_point.endswith("-return"):
                        event_enter = event.probe_point.replace("-return", "-enter")
                        if event_enter in start_events and start_events[event_enter] is not None:
                            if event_enter + ":" + event.probe_point not in self.intervals_time:
                                self.intervals_time[event_enter + ":" + event.probe_point] = []
                            self.intervals_time[event_enter + ":" + event.probe_point].append(event.timestamp - start_events[event_enter])
                            start_events[event_enter] = None
                    else:
                        print("ERROR: Unknown event ending " + str(event.probe_point))
        else:
            print("ERROR: Mode " + str(mode) + " is unknown, aborting.")
            sys.exit()

        for event in experiment.events:
            try:
                self.event_count[event.probe_point]
            except KeyError:
                self.event_count[event.probe_point] = 0
            self.event_count[event.probe_point] += 1
        
        # Remove events from experiment to avoid that python runs out of RAM
        experiment.events = []
        self.experiment = experiment

    def __fill_events_per_process(self, processes):
        events_per_process = {}
        for pid, events in processes.items():
            events_per_process[pid] = len(events)
        return events_per_process

    def check_events_per_process(self, processes) -> bool:
        events_per_process = self.__fill_events_per_process(processes)
        median = np.median(list(events_per_process.values()))
        deviation = np.percentile(list(events_per_process.values()), 75) - np.percentile(list(events_per_process.values()), 25)

        problem = False
        i = 0
        for pid, events in events_per_process.items():
            i += 1
            if np.abs(events - median) > (5 * deviation):
                print(f"{i}: PID {pid}'s number of events ({events}) differs from the median ({median}) more than 5x the deviation ({deviation}).")
                problem = True
        return problem

    def sanity_check(self, processes):
        problem = self.check_events_per_process(processes)
        if problem:
            print("#############################################################")
            print("THIS DATASET DID NOT PASS THE SANITY CHECK! IT MAY MISS DATA!")
            print("#############################################################")

    def export_intervals_to_csv(self, out):
        for name, interv in self.intervals_time.items():
            for cnt, delta in enumerate(interv):
                out.write(str(self.filename) + ',' + str(name) + ',' + str(cnt) + ',' + str(delta) + ',' + str(self.experiment.label) + ',' + str(self.experiment.description) + '\n')

def export_all_intervals_to_csv(ivs, mode):
    with open('intervals_' + mode + '.csv', 'w+') as out:
        out.write('filename,name,id,delta,label,description\n')
        for label, ivs_dict in ivs.items():
            for description, iv_list in ivs_dict.items():
                for iv in iv_list:
                    iv.export_intervals_to_csv(out)

def export_all_intervals_to_pca(ivs, num_q):
    quantiles = np.linspace(0, 1 - 1 / (num_q + 1), (num_q + 1))[1:]
    names = set()
    for label, ivs_dict in ivs.items():
        for description, iv_list in ivs_dict.items():
            for iv in iv_list:
                names.update(list(iv.intervals_time))
    names = list(names)
    names.sort()
    with open('pca.csv', 'w+') as out:
        s = ""
        for name in names:
            for q in quantiles:
                s += name 
                if num_q > 1:
                    s += "-" + str(q)
                s += ","
        out.write('label,description,' + s[:-1] + '\n')
        for label, ivs_dict in ivs.items():
            for description, iv_list in ivs_dict.items():
                for iv in iv_list:
                    s = ""
                    for name in names:
                        if name in iv.intervals_time:
                            for q in np.quantile(iv.intervals_time[name], quantiles):
                                s += "," + str(q)
                        else:
                            s += ",NA" * len(quantiles)
                    out.write(label + "," + description + s + "\n")
                    
def get_fone(tp, fn, tn, fp):
    # Compute the F1 score based on detected samples
    if tp + fp + fn == 0:
        return 'inf'
    return tp / (tp + 0.5 * (fp + fn))

def compute_results(do_print, name, tp, fn, tn, fp, threshold, det_time):
    # Compute metrics and return a dictionary with results
    if tp + fn == 0:
        tpr = 0 # "inf"
    else:
        tpr = tp / (tp + fn)
    if fp + tn == 0:
        fpr = 0 # "inf"
    else:
        fpr = fp / (fp + tn)
    if tn + fp == 0:
        tnr = 0 # "inf"
    else:
        tnr = tn / (tn + fp)
    if tp + fp == 0:
        p = 0 # "inf"
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

def print_confusion(name, tp_c, fn_c, tn_c, fp_c, run, grouping, approach, out_c):
    print(name)
    print("Predicted")
    s = ""
    for description in list(tp_c.values())[0]: # Iterate over the inner dict of a random key from tp_c
        s += description + "\t"
        if len(description) < 8:
            # Extra tab for short descriptions
            s += "\t"
    print(s + "\n" + str("Pos\tNeg\t" * len(tp_c)))
    for description_train, inner_dict in tp_c.items():
        s = ""
        for description in inner_dict:
            s += str(tp_c[description_train][description]) + "\t" + str(fn_c[description_train][description]) + "\t"
            out_c.write(str(run) + "," + approach + "," + str(grouping) + "," + description_train + ",Pos" + "," + description + ",Pos" + "," + str(tp_c[description_train][description]) + "\n")
            out_c.write(str(run) + "," + approach + "," + str(grouping) + "," + description_train + ",Neg" + "," + description + ",Pos" + "," + str(fn_c[description_train][description]) + "\n")
            out_c.write(str(run) + "," + approach + "," + str(grouping) + "," + description_train + ",Pos" + "," + description + ",Neg" + "," + str(fp_c[description_train][description]) + "\n")
            out_c.write(str(run) + "," + approach + "," + str(grouping) + "," + description_train + ",Neg" + "," + description + ",Neg" + "," + str(tn_c[description_train][description]) + "\n")
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

def get_crits(train_mean, train_var, train_cov_inv, train_num_obs, test_vals, test_num_obs, quantiles):
    crits = {}
    for name in train_mean:
        if name not in test_vals:
            continue
        if train_num_obs[name] < len(quantiles):
            continue
        if test_num_obs[name] < len(quantiles):
            continue
        crits[name] = {}
        diff = test_vals[name] - train_mean[name]
        mhd_sq = np.sum(diff @ train_cov_inv[name] * diff)
        p_value = 1 - chi2.cdf(mhd_sq, df=len(quantiles))
        crits[name] = p_value
    return crits

def get_stats(data, num_batches):
    mean = {}
    var = {}
    cov_inv = {}
    for name, vals in data.items():
        if len(vals) < num_batches / 2:
            # Name only appears in some of the batches; likely a result of incorrectly collected intervals
            continue
        mean[name] = np.mean(vals, axis=0)
        var[name] = np.var(vals, axis=0, ddof=0)
        cov = np.cov(vals, rowvar=False)
        try:
            # Compute inverse covariance matrix
            cov_inv[name] = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            try:
                # If that fails, insert small values to avoid singular matrix
                cov += 1e-6 * np.eye(cov.shape[0])
                cov_inv[name] = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # If all fails, try alternative method to invert covariance matrix
                cov_inv[name] = np.linalg.pinv(cov)
    return mean, var, cov_inv

def get_crits_ann(model, center, std, feature_names, test_batch):
    model.eval()
    crits = {}
    with torch.no_grad():
        feature_tensor_list = intervals_to_tensors([test_batch], feature_names)
        embeddings = model(feature_tensor_list)
        distances = torch.norm(embeddings - center, dim=1)
        crits["ann"] = 1.0 / distances.tolist()[0]
    return crits

def intervals_to_tensors(batches, feature_names):
    # Collect delta times as tensors for neural network
    feature_tensor_list = []
    for batch in batches:
        feature_tensors = []
        for name in feature_names:
            tensor = None
            if name in batch.intervals_time:
                data = batch.intervals_time[name]
                tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
            else:
                tensor = torch.empty((0, 1), dtype=torch.float32)
            feature_tensors.append(tensor)
        feature_tensor_list.append(feature_tensors)
    return feature_tensor_list

def get_model_ann(batches, feature_names):
    feature_tensor_list = intervals_to_tensors(batches, feature_names)
    # Create and train neural network with normal batches
    model = DeepSetsEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    epochs = 50
    for epoch in tqdm(range(1, epochs + 1)):
        total_loss = 0
        optimizer.zero_grad()
        embeddings = model(feature_tensor_list)
        center = embeddings.mean(dim=0)
        distances = torch.norm(embeddings - center, dim=1)
        loss = distances.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        tqdm.write(f"Epoch {epoch}: Avg Embedding Spread Loss = {total_loss / 100:.4f}")
    return model

def run_supervised(train, test, quantiles, run, grouping, approach, out_best, out_all, out_c):
    train_num_obs = {}
    train_mean = {}
    train_var = {}
    train_cov_inv = {}
    descriptions = set()
    for label, train_dict in train.items():
        train_num_obs[label] = {}
        train_mean[label] = {}
        train_var[label] = {}
        train_cov_inv[label] = {}
        for description, train_batches in train_dict.items():
            descriptions.add(description)
            train_vals, train_num_obs[label][description] = get_quantile_vals(train_batches, quantiles)
            train_mean[label][description], train_var[label][description], train_cov_inv[label][description] = get_stats(train_vals, len(train_batches))
    pred_pos_act_pos, pred_neg_act_pos, pred_neg_act_neg, pred_pos_act_neg = {}, {}, {}, {}
    tp, fp, tn, fn = {}, {}, {}, {}
    start_time = time.time()
    for d_pred in descriptions:
        pred_pos_act_pos[d_pred] = {}
        pred_neg_act_pos[d_pred] = {}
        pred_neg_act_neg[d_pred] = {}
        pred_pos_act_neg[d_pred] = {}
        tp[d_pred] = {}
        fp[d_pred] = {}
        tn[d_pred] = {}
        fn[d_pred] = {}
        for d_act in descriptions:
            pred_pos_act_pos[d_pred][d_act] = 0
            pred_neg_act_pos[d_pred][d_act] = 0
            pred_neg_act_neg[d_pred][d_act] = 0
            pred_pos_act_neg[d_pred][d_act] = 0
        for label in [rootkit_key, normal_key]:
            tp[d_pred][label] = 0
            fp[d_pred][label] = 0
            tn[d_pred][label] = 0
            fn[d_pred][label] = 0
    for label_act in test:
        for d_act, test_batches in test[label_act].items():
            for i, test_batch in enumerate(test_batches):
                test_vals, test_num_obs = get_quantile_vals([test_batch], quantiles)
                best_label = None
                best_description = None
                highest_pv = -1
                for label_train, _ in train_mean.items():
                    for description_train in train_mean[label_train]:
                        pv_dict = get_crits(train_mean[label_train][description_train], train_var[label_train][description_train], train_cov_inv[label_train][description_train], train_num_obs[label_train][description_train], test_vals, test_num_obs, quantiles)
                        critical_value = sum(list(pv_dict.values()))
                        if critical_value > highest_pv:
                            best_label = label_train
                            best_description = description_train
                            highest_pv = critical_value
                if label_act == best_label and d_act == best_description:
                    # Correct classification; add +1 to classified description/label for TP and +1 to all other description/label for TN
                    for eval_description in descriptions:
                        for eval_label in [rootkit_key, normal_key]:
                            if eval_description == d_act and eval_label == label_act:
                                tp[eval_description][eval_label] += 1
                            else:
                                tn[eval_description][eval_label] += 1
                else:
                    # Incorrect classification; add +1 to misclassified actual value for FN, +1 for incorrect predicted value for FP, and +1 to all other description/label for TN
                    for eval_description in descriptions:
                        for eval_label in [rootkit_key, normal_key]:
                            if eval_description == d_act and eval_label == label_act:
                                fn[eval_description][eval_label] += 1
                            elif eval_description == best_description and eval_label == best_label:
                                fp[eval_description][eval_label] += 1
                            else:
                                tn[eval_description][eval_label] += 1
                # Fill in confusion matrix
                if label_act == rootkit_key and best_label == rootkit_key:
                    pred_pos_act_pos[d_act][best_description] += 1
                elif label_act == rootkit_key and best_label == normal_key:
                    pred_neg_act_pos[d_act][best_description] += 1
                elif label_act == normal_key and best_label == normal_key:
                    pred_neg_act_neg[d_act][best_description] += 1
                elif label_act == normal_key and best_label == rootkit_key:
                    pred_pos_act_neg[d_act][best_description] += 1
                else:
                    print("Labels " + label_act + " or " + best_label + " not one of [" + rootkit_key + ", " + normal_key + "]")
    used_time = time.time() - start_time
    avg = {}
    for description in descriptions:
        for label in [rootkit_key, normal_key]:
            res = compute_results(True, description + "/" + label + " Results (Run " + str(run) + ")", tp[description][label], fn[description][label], tn[description][label], fp[description][label], -1, used_time)
            out_all.write(str(run) + "," + approach + "," + str(grouping) + "," + description + "," + label + "," + str(res["f1"]) + "," + str(res["tp"]) + "," + str(res["fp"]) + "," + str(res["tn"]) + "," + str(res["fn"]) + "," + str(res["time"]) + "," + str(len(quantiles)) + "," + str(res["threshold"]) + "," + str(res["tpr"]) + "," + str(res["fpr"]) + "," + str(res["tnr"]) + "," + str(res["p"]) + "," + str(res["acc"]) + "\n")
            for metric, val in res.items():
                if metric not in avg:
                    avg[metric] = []
                avg[metric].append(val)
    print_confusion("Confusion Matrix (Run " + str(run) + ")", pred_pos_act_pos, pred_neg_act_pos, pred_neg_act_neg, pred_pos_act_neg, run, grouping, approach, out_c)
    out_best.write(str(run) + "," + approach + "," + str(grouping) + "," + str(np.mean(avg["f1"])) + "," + str(np.mean(avg["tp"])) + "," + str(np.mean(avg["fp"])) + "," + str(np.mean(avg["tn"])) + "," + str(np.mean(avg["fn"])) + "," + str(np.mean(avg["time"])) + "," + str(len(quantiles)) + "," + str(np.mean(avg["threshold"])) + "," + str(np.mean(avg["tpr"])) + "," + str(np.mean(avg["fpr"])) + "," + str(np.mean(avg["tnr"])) + "," + str(np.mean(avg["p"])) + "," + str(np.mean(avg["acc"])) + "\n")
    return avg
   
def run_online(ivs, processing_order, num_train, quantiles, run, grouping, approach, out_all, out_best, out_detail):
    train_batches = []
    step = 0
    crits = {}
    for description in processing_order:
        for label in [normal_key, rootkit_key]:
            if label not in crits:
                crits[label] = []
            steps_since_label_change = 0
            for batch in ivs[label][description]:
                step += 1
                steps_since_label_change += 1
                if len(train_batches) < num_train:
                    # For the first few batches, fill up the list of training batches
                    train_batches.append(batch)
                    out_detail.write(str(run) + "," + approach + "," + str(grouping) + "," + str(batch.timestamp) + "," + str(step) + "," + str(steps_since_label_change) + "," + str(label) + "," + str(description) + ",training_dummy," + str(len(quantiles)) + ",1\n")
                    continue
                # Test the current batch against the current list of training batches
                train_vals, train_num_obs = get_quantile_vals(train_batches, quantiles)
                train_mean, train_var, train_cov_inv = get_stats(train_vals, len(train_batches))
                test_vals, test_num_obs = get_quantile_vals([batch], quantiles)
                pv_dict = get_crits(train_mean, train_var, train_cov_inv, train_num_obs, test_vals, test_num_obs, quantiles)
                for name, pv in pv_dict.items():
                    out_detail.write(str(run) + "," + approach + "," + str(grouping) + "," + str(batch.timestamp) + "," + str(step) + "," + str(steps_since_label_change) + "," + str(label) + "," + str(description) + "," + str(name) + "," + str(len(quantiles)) + "," + str(pv) + "\n")
                if steps_since_label_change == 1:
                    # This batch is the first one with a new label; anomaly is expected
                    crits[rootkit_key].append(pv_dict)
                elif steps_since_label_change < num_train:
                    # Training data consist of both normal and anomalous batches; do not change counts for this case
                    pass
                else:
                    # Training data only consists of normal batches; no anomaly is expected
                    crits[normal_key].append(pv_dict)
                # Add batch to the list and remove oldest batch
                train_batches = train_batches[1:] + [batch]
    best_metrics = {"fone": None, "tp": None, "fp": None, "tn": None, "fn": None, "time": None, "thresh": None, "name_counts": None}
    for thresh in np.logspace(-30, 0, num=100):
        start_time = time.time()
        tp, fp, tn, fn = 0, 0, 0, 0
        for label in crits:
            for crit_dict in crits[label]:
                anomaly_detected = False
                for name, crit in crit_dict.items():
                    if crit < thresh:
                        anomaly_detected = True
                        break
                if anomaly_detected:
                    # Detected as anomaly
                    if label == rootkit_key:
                        tp += 1
                    else:
                        fp += 1
                else:
                    # Detected as normal
                    if label == rootkit_key:
                        fn += 1
                    else:
                        tn += 1
        fone = get_fone(tp, fn, tn, fp)
        res_tmp = compute_results(False, "not_print", tp, fn, tn, fp, thresh, -1)
        out_all.write(str(run) + "," + approach + "," + str(grouping) + "," + str(fone) + "," + str(tp) + "," + str(fp) + "," + str(tn) + "," + str(fn) + "," + str(-1) + "," + str(len(quantiles)) + "," + str(thresh) + "," + str(res_tmp["tpr"]) + "," + str(res_tmp["fpr"]) + "," + str(res_tmp["tnr"]) + "," + str(res_tmp["p"]) + "," + str(res_tmp["acc"]) + "\n")
        total_time = time.time() - start_time
        if best_metrics["fone"] is None or fone > best_metrics["fone"]:
            best_metrics["fone"] = fone
            best_metrics["tp"] = tp
            best_metrics["fp"] = fp
            best_metrics["tn"] = tn
            best_metrics["fn"] = fn
            best_metrics["time"] = total_time
            best_metrics["thresh"] = thresh
    res = compute_results(True, "Results (Run " + str(run) + ")", best_metrics["tp"], best_metrics["fn"], best_metrics["tn"], best_metrics["fp"], best_metrics["thresh"], best_metrics["time"])
    out_best.write(str(run) + "," + approach + "," + str(grouping) + "," + str(res["f1"]) + "," + str(res["tp"]) + "," + str(res["fp"]) + "," + str(res["tn"]) + "," + str(res["fn"]) + "," + str(res["time"]) + "," + str(len(quantiles)) + "," + str(res["threshold"]) + "," + str(res["tpr"]) + "," + str(res["fpr"]) + "," + str(res["tnr"]) + "," + str(res["p"]) + "," + str(res["acc"]) + "\n")
    return best_metrics

def run_offline(train, test, quantiles, run, grouping, approach, out_best, out_all, out_c):
    if approach == "ann":
        # ANN Detection
        threshold_search_space = np.logspace(-30, 0, num=100)
        opt_size = 0.5 # Fraction of training data to be used for center/std computation
        model = {}
        feature_names_dict = {}
        centers = {}
        stds = {}
        for description, train_batches in train.items():
            train_split = round(len(train_batches) * (1 - opt_size))
            train_batches_init = train_batches[:train_split]
            train_batches_opt = train_batches[train_split:]
            # Collect feature names so that they are always in order in the next steps
            feature_names = []
            for batch in train_batches_init:
                for name in batch.intervals_time:
                    feature_names.append(name)
            feature_names_dict[description] = feature_names
            # Compute model with initial batches
            model[description] = get_model_ann(train_batches_init, feature_names)
            with torch.no_grad():
                # Compute model statistics with opt batches
                normal_embeddings = model[description](intervals_to_tensors(train_batches_opt, feature_names))
                centers[description] = normal_embeddings.mean(dim=0)
                stds[description] = normal_embeddings.std(dim=0)
        crits = {}
        for label in test:
            print(label)
            for description_train in model:
                # Interate through all training models
                for description, test_batches in test[label].items():
                    # For each training model, iterate through all test values
                    if label not in crits:
                        crits[label] = {}
                    if description_train not in crits[label]:
                        crits[label][description_train] = {}
                    if description not in crits[label][description_train]:
                        crits[label][description_train][description] = []
                    for i, test_batch in enumerate(test_batches):
                        crits[label][description_train][description].append(get_crits_ann(model[description_train], centers[description], stds[description], feature_names_dict[description_train], test_batch))
    else:
        # Shift detection
        threshold_search_space = np.logspace(-30, 0, num=100)
        train_num_obs = {}
        train_mean = {}
        train_var = {}
        train_cov_inv = {}
        for description, train_batches in train.items():
            train_vals, train_num_obs[description] = get_quantile_vals(train_batches, quantiles)
            train_mean[description], train_var[description], train_cov_inv[description] = get_stats(train_vals, len(train_batches))
        crits = {}
        for label in test:
            for description_train in train_mean:
                # Interate through all training models
                for description, test_batches in test[label].items():
                    # For each training model, iterate through all test values
                    if label not in crits:
                        crits[label] = {}
                    if description_train not in crits[label]:
                        crits[label][description_train] = {}
                    if description not in crits[label][description_train]:
                        crits[label][description_train][description] = []
                    for i, test_batch in enumerate(test_batches):
                        test_vals, test_num_obs = get_quantile_vals([test_batch], quantiles)
                        crits[label][description_train][description].append(get_crits(train_mean[description_train], train_var[description_train], train_cov_inv[description_train], train_num_obs[description_train], test_vals, test_num_obs, quantiles))
    best_metrics = {"fone": None, "tp": None, "fp": None, "tn": None, "fn": None, "time": None, "thresh": None, "name_counts": None}
    for thresh in threshold_search_space:
        start_time = time.time()
        tp, fp, tn, fn = 0, 0, 0, 0 # Counts differentiate only normal and anomalous classes, independent from sub-classes
        tp_c, fp_c, tn_c, fn_c = {}, {}, {}, {} # Use sub-classes for the confusion matrix
        name_counts = {} # Counts which function pairs are the ones most often reporting anomalies
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
        out_all.write(str(run) + "," + approach + "," + str(grouping) + "," + str(fone) + "," + str(tp) + "," + str(fp) + "," + str(tn) + "," + str(fn) + "," + str(-1) + "," + str(len(quantiles)) + "," + str(thresh) + "," + str(res_tmp["tpr"]) + "," + str(res_tmp["fpr"]) + "," + str(res_tmp["tnr"]) + "," + str(res_tmp["p"]) + "," + str(res_tmp["acc"]) + "\n")
        total_time = time.time() - start_time
        if best_metrics["fone"] is None or fone > best_metrics["fone"]:
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
    print_confusion("Confusion Matrix (Run " + str(run) + ")", best_metrics["tp_c"], best_metrics["fn_c"], best_metrics["tn_c"], best_metrics["fp_c"], run, grouping, approach, out_c)
    out_best.write(str(run) + "," + approach + "," + str(grouping) + "," + str(res["f1"]) + "," + str(res["tp"]) + "," + str(res["fp"]) + "," + str(res["tn"]) + "," + str(res["fn"]) + "," + str(res["time"]) + "," + str(len(quantiles)) + "," + str(res["threshold"]) + "," + str(res["tpr"]) + "," + str(res["fpr"]) + "," + str(res["tnr"]) + "," + str(res["p"]) + "," + str(res["acc"]) + "\n")
    if False:
        print("Function pairs that reported most anomalies:")
        for label, name_counts_dict in best_metrics["name_counts"].items():
            print(label)
            for name, cnt in name_counts_dict.items():
                print("  " + name + ": " + str(cnt))
    return best_metrics

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
        iv = Intervals(filepath, args.directory, args.grouping)
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

if args.export_intervals:
    export_all_intervals_to_csv(ivs, args.grouping)
    # Use the following command to store intervals for PCA: export_all_intervals_to_pca(ivs, args.quantiles)

if args.mode == "offline":
    with open("results_offline_best_" + args.grouping + ".csv", "w+") as out_best, open("results_offline_all_" + args.grouping + ".csv", "w+") as out_all, open("results_offline_confusion_" + args.grouping + ".csv", "w+") as out_c:
        out_best.write("run,approach,group,fone,tp,fp,tn,fn,time,q,thresh,tpr,fpr,tnr,p,acc\n")
        out_all.write("run,approach,group,fone,tp,fp,tn,fn,time,q,thresh,tpr,fpr,tnr,p,acc\n")
        out_c.write("run,approach,group,pred,pred_class,actual,actual_class,cnt\n")
        for run in range(args.repeat):
            run += 1 # Start with run #1
            # Get training data from normal data (default case) and remove it from test data
            ivs_train = {}
            for description in ivs[normal_key]:
                if args.quantiles > 0:
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
                quantiles = np.linspace(0, 1 - 1 / (args.quantiles + 1), (args.quantiles + 1))[1:] # Excludes 1 to avoid last term (which is usually an outlier), e.g., for args.quantiles = 100 will result in 0, 0.01, 0.02, ..., 0.99
            else:
                # This case is just used to test teh influence of the number of quantiles
                quantiles = np.linspace(0, 1 - 1 / (run + 1), (run + 1))[1:] # Increase the number of quantiles by 1 in every run
            best_metrics = run_offline(ivs_train, ivs, quantiles, run, args.grouping, args.approach, out_best, out_all, out_c)
        
            # Return training data to normal data in case that there is another iteration
            for description in ivs_train:
                ivs[normal_key][description].extend(ivs_train[description])
elif args.mode == "online":
    with open("results_online_detail_" + args.grouping + ".csv", "w+") as out_detail, open("results_online_best_" + args.grouping + ".csv", "w+") as out_best, open("results_online_all_" + args.grouping + ".csv", "w+") as out_all:
        out_detail.write("run,approach,group,ts,step,anom_step,label,description,name,q,pv\n")
        out_best.write("run,approach,group,fone,tp,fp,tn,fn,time,q,thresh,tpr,fpr,tnr,p,acc\n")
        out_all.write("run,approach,group,fone,tp,fp,tn,fn,time,q,thresh,tpr,fpr,tnr,p,acc\n")
        for run in range(args.repeat):
            run += 1 # Start with run #1
            processing_order = list(list(ivs.values())[0])
            if run > 1:
                # Leave batches in order for the first run, then shuffle the descriptions (not labels!) in each subsequent run
                random.shuffle(processing_order)
            num_train = math.ceil(len(ivs[normal_key][processing_order[0]]) * args.train_ratio)
            print("Data will be processed with a sliding window of size " + str(num_train) + " in the following order:")
            for processing_step in processing_order:
                print("  " + processing_step + " " + normal_key + ": " + str(len(ivs[normal_key][processing_step])) + " batches")
                print("  " + processing_step + " " + rootkit_key + ": " + str(len(ivs[rootkit_key][processing_step])) + " batches")
            
            if args.quantiles > 0:
                quantiles = np.linspace(0, 1 - 1 / (args.quantiles + 1), (args.quantiles + 1))[1:] # Excludes 1 to avoid last term (which is usually an outlier), e.g., for args.quantiles = 100 will result in 0, 0.01, 0.02, ..., 0.99
            else:
                # This case is just used to test the influence of the number of quantiles
                quantiles = np.linspace(0, 1 - 1 / (run + 1), (run + 1))[1:] # Increase the number of quantiles by 1 in every run
            run_online(ivs, processing_order, num_train, quantiles, run, args.grouping, args.approach, out_all, out_best, out_detail)
elif args.mode == "supervised":
    # Be aware that this mode is experimental and does not yield good results
    with open("results_supervised_best_" + args.grouping + ".csv", "w+") as out_best, open("results_supervised_all_" + args.grouping + ".csv", "w+") as out_all, open("results_supervised_confusion_" + args.grouping + ".csv", "w+") as out_c:
        out_best.write("run,approach,group,fone,tp,fp,tn,fn,time,q,thresh,tpr,fpr,tnr,p,acc\n")
        out_all.write("run,approach,group,description,label,fone,tp,fp,tn,fn,time,q,thresh,tpr,fpr,tnr,p,acc\n")
        out_c.write("run,approach,group,pred,pred_class,actual,actual_class,cnt\n")
        for run in range(args.repeat):
            run += 1 # Start with run #1
            ivs_train = {}
            for label in ivs:
                ivs_train[label] = {}
                for description in ivs[label]:
                    if args.quantiles > 0:
                        random.shuffle(ivs[label][description])
                    split_point = math.ceil(len(ivs[label][description]) * args.train_ratio)
                    ivs_train[label][description] = ivs[label][description][:split_point]
                    ivs[label][description] = ivs[label][description][split_point:]
            print("Batches for training: " + str(sum(len(value) for value in ivs_train[normal_key].values()) + sum(len(value) for value in ivs_train[rootkit_key].values())))
            print("  Normal batches: " + str(sum(len(value) for value in ivs_train[normal_key].values())))
            for description in ivs_train[normal_key]:
                print("    " + description + ": " + str(len(ivs_train[normal_key][description])))
            print("  Anomalous batches: "  + str(sum(len(value) for value in ivs_train[rootkit_key].values())))
            for description in ivs_train[rootkit_key]:
                print("    " + description + ": " + str(len(ivs_train[rootkit_key][description])))
            print("Batches for testing: " + str(sum(len(value) for value in ivs[normal_key].values()) + sum(len(value) for value in ivs[rootkit_key].values())))
            print("  Normal batches: " + str(sum(len(value) for value in ivs[normal_key].values())))
            for description in ivs[normal_key]:
                print("    " + description + ": " + str(len(ivs[normal_key][description])))
            print("  Anomalous batches: "  + str(sum(len(value) for value in ivs[rootkit_key].values())))
            for description in ivs[rootkit_key]:
                print("    " + description + ": " + str(len(ivs[rootkit_key][description])))
            
            if args.quantiles > 0:
                quantiles = np.linspace(0, 1 - 1 / (args.quantiles + 1), (args.quantiles + 1))[1:] # Excludes 1 to avoid last term (which is usually an outlier), e.g., for args.quantiles = 100 will result in 0, 0.01, 0.02, ..., 0.99
            else:
                # This case is just used to test the influence of the number of quantiles
                quantiles = np.linspace(0, 1 - 1 / (run + 1), (run + 1))[1:] # Increase the number of quantiles by 1 in every run
            best_metrics = run_supervised(ivs_train, ivs, quantiles, run, args.grouping, args.approach, out_best, out_all, out_c)

            # Return training data to normal data in case that there is another iteration
            for label in ivs:
                for description in ivs[label]:
                    ivs[label][description].extend(ivs_train[label][description])
