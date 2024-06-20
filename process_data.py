import json
from itertools import chain

import matplotlib.pyplot as plt
from multiprocessing import Process
import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import pandasql as psql
from data_classes import Event, Interval, experiment_from_json


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
    experiment = None
    events = []
    events_per_process = {}
    dataframe = None

    def __init__(self, args):
        self.args = args
        try:
            filename = args[1]
        except IndexError:
            # get the most recent output file...
            files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
            outputs = [file for file in files if re.match(r'^output.*\.json$', file)]
            outputs.sort()
            filename = outputs.pop()

        with open(filename, 'r') as file:
            json_obj = json.load(file)
            experiment = experiment_from_json(json_obj)
            self.experiment = experiment
            self.events = experiment.events

        self.file_date = filename.replace("output", "").replace(".json", "")

        for event in experiment.events:
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
