import array
import json
from itertools import chain
from statistics import median
from typing import *

import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Process
import os
import re
import gzip
import math
import numpy as np
import pandas as pd
import pandasql as psql
from skimage.filters import threshold_otsu
from scipy.stats import median_abs_deviation
from data_classes import Event, Interval, experiment_from_json
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

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

        self.file_date = filename.replace("output", "").replace(".json", "").replace(".gz", "")

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

    def distribution_split(self):
        def make_histogram(name: str, values_a: [int], values_b: [int]) -> None:
            means_a = split_gaussian_mixture(name, values_a, "normal")
            means_b = split_gaussian_mixture(name, values_b, "with rootkit")
            plt.hist(values_a, bins=__unique_vals__(values_a), label="normal")
            plt.hist(values_b, bins=__unique_vals__(values_b), label="with rootkit")
            for mean_a in means_a:
                plt.axvline(mean_a, color='blue', linestyle='dashed', linewidth=1)
            for mean_b in means_b:
                plt.axvline(mean_b, color='red', linestyle='dotted', linewidth=1)
            plt.xlim(0, max(means_a + means_b) * 2)
            plt.title(name)
            plt.yscale('log')
            plt.legend()
            plt.tight_layout()
            plt.savefig("distribution_split_" + self.file_date + "_" + name + ".svg")
            print(name + " saved")
            plt.clf()

        def split_gaussian_mixture(name: str, values: [int], rk) -> None:
            data = np.array(values)
            data = data.reshape(-1, 1)
            n_components_range = range(1, 11)
            bics = []
            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(data)
                bics.append(gmm.bic(data))

            # Find the optimal number of components
            optimal_n_components = n_components_range[np.argmin(bics)]
            #print(f"Optimal number of components: {optimal_n_components}")

            # Fit the final GMM with the optimal number of components
            gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
            gmm.fit(data)

            # Get the means, covariances, and weights of the GMM components
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            weights = gmm.weights_

            # Filter out components with weights below the threshold
            weight_threshold = 0.001 # Components with less than that of the total data will be discarded
            var_threshold = 80000000 # Components with higher variances will be discarded
            valid_components = (weights > weight_threshold) & (covariances < var_threshold)

            # Print the parameters of the valid components
            return_means = []
            print(name + " " + rk)
            for i, valid in enumerate(valid_components):
                if valid:
                    return_means.append(means[i])
                    #print(f"Component {i + 1}:")
                    print(f"  Mean: {means[i]:.2f}")
                    print(f"  Variance: {covariances[i]:.2f}")
                    print(f"  STD: {math.sqrt(covariances[i]):.2f}")
                    print(f"  Weight: {weights[i]:.5f}")
            return return_means

        workers = []
        for name in ["filldir64-return:filldir64-enter"]:  # self.interval_types
            try:
                worker = Process(target=make_histogram, args=[name, [x.time for x in self.intervals[name]], [x.time for x in self.intervals_rootkit[name]]])
                worker.start()
                workers.append(worker)
            except KeyError:
                pass
        for worker in workers:
            worker.join()

    def split(self, interval: str):
        data_reference = [interval.time for interval in self.intervals[interval]]
        data_rootkit = [interval.time for interval in self.intervals_rootkit[interval]]
        #plt.boxplot(data, labels=["without rootkit", "rootkitted"], patch_artist=True)

        upper_cut = np.mean(data_reference) * 5
        data_reference = [i for i in data_reference if i < upper_cut]

        data_reference = np.array(data_reference).reshape(-1, 1)

        gmm = GaussianMixture(n_components=10, covariance_type='full')
        gmm.fit(data_reference)

        labels = gmm.predict(data_reference)

        plt.yscale('log')
        plt.xlim()

        components = np.unique(labels)

        for i, component in enumerate(components):
            component_data = data_reference[labels == component]
            plt.hist(component_data, bins=int(len(component_data)/10))

        plt.savefig("fooo" + self.file_date + ".svg")

    def kmeans(self, interval: str):
        data_reference = [interval.time for interval in self.intervals[interval]]
        data_rootkit = [interval.time for interval in self.intervals_rootkit[interval]]
        # plt.boxplot(data, labels=["without rootkit", "rootkitted"], patch_artist=True)

        upper_cut = np.mean(data_reference) * 5
        data_reference = [i for i in data_reference if i < upper_cut]

        data = pd.DataFrame(data_reference, columns=['runtime'])

        k = 10

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

        data['cluster'] = kmeans.fit_predict(data[['runtime']])

        for i in range(k):
            plt.hist(data[data['cluster'] == i]['runtime'],
                     bins=15, alpha=0.6, label=f'Cluster {i + 1}', edgecolor='black')

        plt.title('Histogram of Runtimes by Cluster')
        plt.yscale('log')
        plt.xlabel('Runtime (ns)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid()
        plt.savefig("baaar" + self.file_date + ".svg")

    def distribution_comparison(self, only_interval:Optional[str]=None):
        def make_histogram(name: str, values_a: [int], values_b: [int]) -> None:
            plt.hist(values_a, bins=__unique_vals__(values_a), label="normal")
            plt.hist(values_b, bins=__unique_vals__(values_b), label="with rootkit")
            plt.title(name)
            plt.yscale('log')
            plt.xlabel('nano seconds')
            plt.legend()
            plt.tight_layout()
            plt.savefig("distribution_comparison_" + self.file_date + "_" + name + ".svg")
            print(name + " saved")
            plt.clf()

        workers = []
        for name in self.interval_types if only_interval is None else [only_interval]:
            try:
                data_reference = [x.time for x in self.intervals[name]]
                data_rootkit = [x.time for x in self.intervals_rootkit[name]]
                upper_cut = np.mean(data_reference) * 4
                data_reference = [i for i in data_reference if i < upper_cut]
                data_rootkit = [i for i in data_rootkit if i < upper_cut]
                worker = Process(target=make_histogram, args=[name, data_reference, data_rootkit])
                worker.start()
                workers.append(worker)
            except KeyError:
                pass
        for worker in workers:
            worker.join()

    def export_intervals_to_csv(self):
        with open(self.file_date + '.csv', 'w+') as out:
            out.write('name,id,delta,pid,tgit,class\n')
            for name, interv in self.intervals.items():
                for cnt, i in enumerate(interv):
                    out.write(str(name) + ',' + str(cnt) + ',' + str(i.time) + ',' + str(i.pid) + ',' + str(i.tgid) + ',Normal' + '\n')
            for name, interv in self.intervals_rootkit.items():
                for cnt, i in enumerate(interv):
                    out.write(str(name) + ',' + str(cnt) + ',' + str(i.time) + ',' + str(i.pid) + ',' + str(i.tgid) + ',Anomaly' + '\n')

    def interval_means(self):
        print("####interval means####")
        print(f"{'name'.ljust(55)}\tnormal\t\trootkitted\tpercent slower")
        print("value:standard_deviation #count")
        for name in self.interval_types:
            try:
                data_reference = [i.time for i in self.intervals[name]]
                upper_cut = np.mean(data_reference) * 5
                data_reference = [i for i in data_reference if i < upper_cut]
                data_rootkited = [i.time for i in self.intervals_rootkit[name]]
                data_rootkited = [i for i in data_rootkited if i < upper_cut]
                normal_mean = np.mean(data_reference)
                normal_std = np.std(data_reference)
                rootkit_mean = np.mean(data_rootkited)
                rootkit_std = np.std(data_rootkited)
                factor_mean = (rootkit_mean/normal_mean-1)
                print(f"{name.ljust(55)}\t{normal_mean:.1f}:{normal_std:.1f} #{len(self.intervals[name])}\t\t{rootkit_mean:.1f}:{rootkit_std:.1f} #{len(self.intervals_rootkit[name])}\t\t{factor_mean*100:.1f}")

            except KeyError:
                pass
        print("######################")


    def interval_medians(self):
        print("####interval medians####")
        print(f"{'name'.ljust(55)}\tnormal\t\trootkitted\tpercent slower")
        print("value:median_absolute_dviation #count")
        for name in self.interval_types:
            try:
                normal_median = median([i.time for i in self.intervals[name]])
                normal_mad = median_abs_deviation([i.time for i in self.intervals[name]])
                rootkit_median = median([i.time for i in self.intervals_rootkit[name]])
                rootkit_mad = median_abs_deviation([i.time for i in self.intervals_rootkit[name]])
                factor_median = (rootkit_median/normal_median-1)
                print(f"{name.ljust(55)}{normal_median:.1f}:{normal_mad:.1f} #{len(self.intervals[name])}\t\t{rootkit_median:.1f}:{rootkit_mad:.1f} #{len(self.intervals_rootkit[name])}\t\t{factor_median*100:.1f}")

            except KeyError:
                pass
        print("######################")


    def interval_means_latex(self):
        print("####interval means####")
        print(f"name\tnormal\t\trootkitted\tpercent slower")
        for name in self.interval_types:
            try:
                data_reference = [i.time for i in self.intervals[name]]
                upper_cut = np.mean(data_reference) * 5
                data_reference = [i for i in data_reference if i < upper_cut]
                data_rootkited = [i.time for i in self.intervals_rootkit[name]]
                data_rootkited = [i for i in data_rootkited if i < upper_cut]
                normal_mean = np.mean(data_reference)
                rootkit_mean = np.mean(data_rootkited)
                factor_mean = (rootkit_mean / normal_mean - 1)
                name_escaped = name.replace('_', '\_')
                print(f"{name_escaped} & {normal_mean:.1f} & {rootkit_mean:.1f} & {factor_mean*100:.1f} \\\\")
            except KeyError:
                pass
        print("######################")


    def interval_medians_latex(self):
        print("####interval medians####")
        print(f"{'name'.ljust(55)}\tnormal\t\trootkitted\tpercent slower")
        for name in self.interval_types:
            try:
                normal_median = median([i.time for i in self.intervals[name]])
                rootkit_median = median([i.time for i in self.intervals_rootkit[name]])
                factor_median = (rootkit_median / normal_median - 1)
                name_escaped = name.replace('_', '\_')
                print(f"{name_escaped} & {normal_median:.1f} & {rootkit_median:.1f} & {factor_median*100:.1f} \\\\")

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


    def get_slice_means(self, slices: List[Tuple[int, int]], interval: str) -> List[Tuple[int, int]]:

        data_reference = [x.time for x in self.intervals[interval]]
        data_rootkit = [x.time for x in self.intervals_rootkit[interval]]
        upper_cut = np.mean(data_reference) * 4
        data_reference = [i for i in data_reference if i < upper_cut]
        data_rootkit = [i for i in data_rootkit if i < upper_cut]

        result_slices = []

        for slice in slices:
            begin = slice[0]
            end = slice[1]

            slice_reference = [i for i in data_reference if begin < i < end]
            slice_rootkit = [i for i in data_rootkit if begin < i < end]

            slice_reference_mean = np.mean(slice_reference)
            slice_rootkit_mean = np.mean(slice_rootkit)

            print("mean (reference):\t" + str(slice_reference_mean))
            print("mean (rootkit):\t" + str(slice_rootkit_mean))
            print("difference:\t" + str(slice_rootkit_mean - slice_reference_mean) + "\n")

            result_slices.append((slice_reference_mean, slice_rootkit_mean))

        return result_slices


    def make_result_histogram(self):
        interval_name = "filldir64-return:filldir64-enter"
        color_reference = "tab:orange"
        color_rootkit = "tab:blue"

        data_reference = [x.time for x in self.intervals[interval_name]]
        data_rootkit = [x.time for x in self.intervals_rootkit[interval_name]]
        upper_cut = np.mean(data_reference) * 4
        data_reference = [i for i in data_reference if i < upper_cut]
        data_rootkit = [i for i in data_rootkit if i < upper_cut]

        plt.hist(data_reference, bins=__unique_vals__(data_reference), label="normal", color=color_reference)
        plt.hist(data_rootkit, bins=__unique_vals__(data_rootkit), label="with rootkit", color=color_rootkit)
        plt.title(interval_name)
        plt.yscale('log')
        plt.xlabel('nano seconds')

        slice_means = self.get_slice_means([(0, 1650), (3550, 7000), (7000, 10300), (14840, 17440), (18270, 21470)], "filldir64-return:filldir64-enter")

        for slice_mean in slice_means:
            a = slice_mean[0]
            b = slice_mean[1]
            plt.axvline(x=a, linestyle="--", zorder=1, lw=1, color=color_reference)
            plt.axvline(x=b, linestyle="--", zorder=1, lw=1, color=color_rootkit)
            plt.plot(slice_mean, [100, 100], color='gray', lw=1, linestyle="-")
            plt.annotate("{:.1f}".format(b-a), xy=(a + (b-a)/2, 100), xytext=(b+400, 120), arrowprops=dict(arrowstyle='->', color='black', lw=1))



        plt.legend()
        plt.tight_layout()
        plt.savefig("distribution_comparison_" + self.file_date + "_" + interval_name + ".svg")
        print(interval_name + " saved")
        plt.clf()

    def get_weighted_overal_slice_mean(self, slices: List[Tuple[int, int]], interval: str) -> Tuple[int, int]:

        data_reference = [x.time for x in self.intervals[interval]]
        data_rootkit = [x.time for x in self.intervals_rootkit[interval]]
        upper_cut = np.mean(data_reference) * 4
        data_reference = [i for i in data_reference if i < upper_cut]
        data_rootkit = [i for i in data_rootkit if i < upper_cut]

        data_ref_len = 0
        data_rk_len = 0
        result_ref = 0
        result_rk = 0

        for slice in slices:
            begin = slice[0]
            end = slice[1]

            slice_reference = [i for i in data_reference if begin < i < end]
            slice_rootkit = [i for i in data_rootkit if begin < i < end]

            slice_reference_mean = np.mean(slice_reference)
            slice_rootkit_mean = np.mean(slice_rootkit)

            result_ref += slice_reference_mean * len(slice_reference)
            result_rk += slice_rootkit_mean * len(slice_rootkit)
            data_ref_len += len(slice_reference)
            data_rk_len += len(slice_rootkit)

        result_ref = result_ref / data_ref_len
        result_rk = result_rk / data_rk_len

        print(f"ref: {result_ref:.1f}, rk: {result_rk:.1f}")

        return (result_ref, result_rk)
