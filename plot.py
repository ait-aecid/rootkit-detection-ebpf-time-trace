import json
import matplotlib.pyplot as plt
from multiprocessing import Process
import os
import re
import sys
from data_classes import Event, Interval


def __unique_vals__(lst: list) -> int:
    values = []
    unique = 0
    for elem in lst:
        if elem not in values:
            values.append(elem)
            unique += 1
    return unique


class Plot:
    args = []
    processes = {}
    intervals = {}
    file_date = ""

    def __init__(self, args):
        self.args = args
        try:
            filename = args[1]
        except IndexError:
            # get the most recent output file...
            files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
            outputs = [file for file in files if re.match(r'^output.*\.json$', file)]
            filename = outputs[-1]

        with open(filename, 'r') as file:
            json_obj = json.load(file)
            events = [Event(**elem) for elem in json_obj]

        self.file_date = filename.replace("output", "").replace(".json", "")

        for event in events:
            try:
                self.processes[event.pid]
            except KeyError:
                self.processes[event.pid] = []
            self.processes[event.pid].append(event)

        for pid in self.processes:
            for i in range(len(self.processes[pid]) - 1):
                event_a = self.processes[pid][i]
                event_b = self.processes[pid][i+1]
                type_name = event_a.probe_point + ":" + event_b.probe_point
                try:
                    self.intervals[type_name]
                except KeyError:
                    self.intervals[type_name] = []
                self.intervals[type_name].append(Interval(event_b.timestamp - event_a.timestamp, event_a, event_b, pid, event_a.tgid))

    def interval_type_counts(self):
        plt.barh([x for x in self.intervals.keys()], [len(x) for x in self.intervals.values()])
        plt.tight_layout()
        plt.savefig("interval_distribution_" + self.file_date + ".svg")
        plt.clf()

    def distributions(self):
        def make_histogram(name: str, values: [int]) -> None:
            plt.hist(values, bins=__unique_vals__(values))
            plt.title(name)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig("distribution_" + self.file_date + "_" + name + ".svg")
            print(name + " saved")
            plt.clf()

        plot_processes = []

        for name, values in self.intervals.items():
            worker = Process(target=make_histogram, args=[name, [x.time for x in values]])
            worker.start()
            plot_processes.append(worker)

        for worker in plot_processes:
            worker.join()

    def interval_types_per_run(self):
        # 16

        fig, axs = plt.subplots(4, 4)

        intervals_filtered = {}
        for name, intervals in self.intervals.items():
            for elem in intervals:
                pid = elem.pid
                try:
                    intervals_filtered[pid]
                except KeyError:
                    intervals_filtered[pid] = {}
                try:
                    intervals_filtered[pid][name]
                except KeyError:
                    intervals_filtered[pid][name] = 0
                intervals_filtered[pid][name] += 1

        for i in range(16):
            axs[int(i/4), i % 4].barh(list(intervals_filtered[list(self.processes.keys())[i]].keys()),
                                      intervals_filtered[list(self.processes.keys())[i]].values())
        fig.savefig("interval_types_per_run_" + self.file_date + ".svg")
        fig.clf()
