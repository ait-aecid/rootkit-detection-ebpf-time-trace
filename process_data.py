import json
from itertools import chain

import matplotlib.pyplot as plt
from multiprocessing import Process
import os
import re
import gzip
import numpy as np
import pandas as pd
import pandasql as psql
from skimage.filters import threshold_otsu
from data_classes import Event, Interval, experiment_from_json


def __unique_vals__(lst: list) -> int:
    values = []
    unique = 0
    for elem in lst:
        if elem not in values:
            values.append(elem)
            unique += 1
    return unique


def mean(intervals: [Interval]) -> float:
    return sum([i.time for i in intervals]) / len(intervals)


class Plot:
    args = []
    processes = {}
    processes_rootkit = {}
    intervals = {}
    intervals_rootkit = {}
    interval_types = []
    file_date = ""
    experiment = None
    events = []
    events_rootkit = []
    events_per_process = {}
    events_per_process_rootkit = {}
    dataframe = None
    event_count = {}
    event_count_rootkit = {}

    def __init__(self, args):
        self.args = args
        try:
            filename = args[1]
        except IndexError:
            # get the most recent output file...
            files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
            outputs = [file for file in files if re.match(r'^output.*\.json.gz$', file)]
            outputs.sort()
            filename = outputs.pop()

        with gzip.open(filename, 'r') as file:
            json_obj = json.load(file)
            experiment = experiment_from_json(json_obj)
            self.experiment = experiment
            self.events = experiment.events
            self.events_rootkit = experiment.events_rootkit

        self.file_date = filename.replace("output", "").replace(".json", "")

        for event in experiment.events:
            try:
                self.processes[event.pid]
            except KeyError:
                self.processes[event.pid] = []
            self.processes[event.pid].append(event)

        for event in experiment.events_rootkit:
            try:
                self.processes_rootkit[event.pid]
            except KeyError:
                self.processes_rootkit[event.pid] = []
            self.processes_rootkit[event.pid].append(event)

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

        for pid in self.processes_rootkit:
            for i in range(len(self.processes_rootkit[pid]) - 1):
                event_a = self.processes_rootkit[pid][i]
                event_b = self.processes_rootkit[pid][i + 1]
                type_name = event_a.probe_point + ":" + event_b.probe_point
                try:
                    self.intervals_rootkit[type_name]
                except KeyError:
                    self.intervals_rootkit[type_name] = []
                self.intervals_rootkit[type_name].append(
                    Interval(event_b.timestamp - event_a.timestamp, event_a, event_b, pid, event_a.tgid))

        for interval_type in self.intervals.keys():
            if interval_type not in self.interval_types:
                self.interval_types.append(interval_type)
        for interval_type in self.intervals_rootkit.keys():
            if interval_type not in self.interval_types:
                self.interval_types.append(interval_type)

        for event in self.events:
            try:
                self.event_count[event.probe_point]
            except KeyError:
                self.event_count[event.probe_point] = 0
            self.event_count[event.probe_point] += 1

        for event in self.events_rootkit:
            try:
                self.event_count_rootkit[event.probe_point]
            except KeyError:
                self.event_count_rootkit[event.probe_point] = 0
            self.event_count_rootkit[event.probe_point] += 1

    def interval_type_counts(self):
        plt.barh([x for x in self.intervals.keys()], [len(x) for x in self.intervals.values()])
        plt.tight_layout()
        plt.savefig("interval_distribution_" + self.file_date + ".svg")
        plt.clf()

    def distributions(self):
        def make_histogram(name: str, values: [int]) -> None:
            plt.hist(values, bins=__unique_vals__(values))
            plt.title(name)
            plt.yscale('linear')
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

    def distribution_comparison(self):
        def make_histogram(name: str, values_a: [int], values_b: [int]) -> None:
            plt.hist(values_a, bins=__unique_vals__(values_a), label="normal")
            plt.hist(values_b, bins=__unique_vals__(values_b), label="with rootkit")
            plt.title(name)
            plt.yscale('log')
            plt.legend()
            plt.tight_layout()
            plt.savefig("distribution_comparison_" + self.file_date + "_" + name + ".svg")
            print(name + " saved")
            plt.clf()

        workers = []
        for name in self.interval_types:
            try:
                worker = Process(target=make_histogram, args=[name, [x.time for x in self.intervals[name]], [x.time for x in self.intervals_rootkit[name]]])
                worker.start()
                workers.append(worker)
            except KeyError:
                pass
        for worker in workers:
            worker.join()

    def interval_means(self):
        print("####interval means####")
        print(f"{'name'.ljust(55)}\tnormal\trootkitted\tfactor")
        for name in self.interval_types:
            try:
                normal = mean(self.intervals[name])
                rootkit = mean(self.intervals_rootkit[name])
                print(f"{name.ljust(55)}\t{normal:.1f}\t\t{rootkit:.1f}\t\t{rootkit/normal-1:.3f}")
            except KeyError:
                pass
        print("######################")

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
            axs[int(i / 4), i % 4].barh(list(intervals_filtered[list(self.processes.keys())[i]].keys()),
                                        intervals_filtered[list(self.processes.keys())[i]].values())
        fig.savefig("interval_types_per_run_" + self.file_date + ".svg")
        fig.clf()

    def interval_types_per_run2(self):
        def get_or0(dictionary, outer, inner):
            try:
                return dictionary[outer][inner]
            except KeyError:
                return 0
        intervals_filtered = {}
        for name, intervals in self.intervals.items():
            for elem in intervals:
                pid = elem.pid
                if pid not in list(self.processes.keys()):
                    pass
                try:
                    intervals_filtered[name]
                except KeyError:
                    intervals_filtered[name] = {}
                try:
                    intervals_filtered[name][pid]
                except KeyError:
                    intervals_filtered[name][pid] = 0
                intervals_filtered[name][pid] += 1

        groups = list(intervals_filtered.keys())
        categories = list(intervals_filtered[groups[0]].keys())
        n_groups = len(groups)
        n_categories = len(categories)
        fig, ax = plt.subplots()
        bar_width = 0.05
        index = np.arange(n_groups)
        for i, category in enumerate(categories):
            values = [get_or0(intervals_filtered, group, category) for group in groups]
            bar_positions = index + i * bar_width
            ax.bar(bar_positions, values, bar_width, label=category)
        ax.set_xlabel('Interval Name')
        ax.set_ylabel('Interval Measure Count')
        ax.set_xticks(index + bar_width * (n_categories / 2 - 0.5))
        ax.set_xticklabels(groups, rotation=90)
        plt.tight_layout()

        filename = "interval_types_per_run2_" + self.file_date + ".svg"
        plt.savefig(filename)
        plt.clf()

    def boxplot(self, interval: str):
        data = [[interval.time for interval in self.intervals[interval]], [interval.time for interval in self.intervals_rootkit[interval]]]
        plt.boxplot(data, labels=["without rootkit", "rootkitted"], patch_artist=True)

        plt.title(interval)
        plt.ylabel("nano seconds")

        plt.tight_layout()
        filename = "boxplots_" + self.file_date + ".svg"
        plt.savefig(filename)
        print(f"saved {filename}")
        plt.clf()

    def boxplot4(self):
        interval = "filldir64-return:filldir64-enter"

        data_no_rootkit = self.intervals[interval]
        data_rootkitted = self.intervals_rootkit[interval]

        threshold = threshold_otsu(np.array([i.time for i in data_no_rootkit] + [i.time for i in data_rootkitted]))

        data = [
            [i.time for i in data_no_rootkit if i.time < threshold],
            [i.time for i in data_no_rootkit if i.time >= threshold],
            [i.time for i in data_rootkitted if i.time < threshold],
            [i.time for i in data_rootkitted if i.time >= threshold]
        ]
        plt.boxplot(data, labels=["no rootkit",
                                  "no rootkit",
                                  "rootkitted",
                                  "rootkitted"],
                    patch_artist=True)

        plt.title(interval)
        plt.ylabel("nano seconds")

        print("### interval means [cuttoff] ###")
        print("mean above threshold, no rootkit: " + str(mean([i for i in data_no_rootkit if i.time >= threshold])))
        print("mean above threshold, rootkitted: " + str(mean([i for i in data_rootkitted if i.time >= threshold])))
        print("################################")

        plt.tight_layout()
        filename = "boxplot4_" + self.file_date + ".svg"
        plt.savefig(filename)
        print(f"saved {filename}")
        plt.clf()

    def print_num_processes(self):
        print("# of PIDs found in dataset: " + str(len(self.processes)))

    def __fill_events_per_process(self):
        if self.events_per_process:
            # was already filled
            return
        for pid in self.processes:
            num_events = len([x for x in self.events if x.pid == pid])
            # print(f"In {pid} are {num_events} events.")#
            self.events_per_process[pid] = num_events

    def __generate_pandas_dataframe(self):
        if self.dataframe:
            return
        self.dataframe = pd.DataFrame([vars(event) for event in self.events])

    def print_num_events_per_process(self, event_name: str):
        self.__generate_pandas_dataframe()
        df = self.dataframe
        query = f"SELECT pid, count(*) AS event_count FROM df WHERE probe_point LIKE '%{event_name}%' GROUP BY pid"
        result = psql.sqldf(query, locals())
        print(result)

    def plot_event_timeline(self):
        """
        Plot timeline of single events for the first PID.
        """
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        self.__generate_pandas_dataframe()
        df = self.dataframe
        query = f"SELECT DISTINCT probe_point FROM df"
        df = psql.sqldf(query, locals())
        event_types = list(chain(*df.values.tolist()))
        colors = {}
        counter = 0
        for event_type in event_types:
            colors[event_type] = list(mcolors.BASE_COLORS.values())[counter]
            counter += 1

        middle_pid = list(self.processes.keys())[int(len(list(self.processes.keys())) / 2)]
        first_pid = list(self.processes.keys())[0]
        events = [event for event in self.events if event.pid == first_pid]

        plt.figure(figsize=(10, 2))
        fig, ax = plt.subplots()

        # Scatter plot each event on the timeline
        for event in events:
            ax.scatter(event.timestamp, 0, marker='.', color=colors[event.probe_point], s=2)
            #ax.text(event.timestamp, 0.05, event.probe_point, ha='center')

        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_yticks([])
        ax.set_title('Timeline of Events')
        legend = []
        for name, color in colors.items():
            legend.append(mpatches.Patch(color=color, label=name.replace("__", "")))
            print(name)
        ax.legend(handles=legend)

        filename = "event_timeline2_" + self.file_date + ".svg"
        plt.savefig(filename)
        plt.clf()

    def print_statistics(self):
        print("Dataset has")
        print(f"{len(self.events)} events for the run without rootkit")
        print(f"and {len(self.events_rootkit)} events with rootkit.")
        print("")
        print("no rootkit")
        for key, value in self.event_count.items():
            print(f"{key}: {value}")
        print("rootkit")
        for key, value in self.event_count_rootkit.items():
            print(f"{key}: {value}")
        print("")


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
