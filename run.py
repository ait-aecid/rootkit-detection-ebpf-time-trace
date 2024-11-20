import sys
from process_data import Plot

try:
    filename = sys.argv[1]
except IndexError:
    # get the most recent experiment file...
    files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
    outputs = [file for file in files if re.match(r'^experiment.*\.json.gz$', file)]
    outputs.sort()
    filename = outputs.pop()

plot = Plot(filename)

plot.sanity_check()


#plot.split("filldir64-return:filldir64-enter")
#plot.kmeans("filldir64-return:filldir64-enter")
plot.distribution_comparison("filldir64-return:filldir64-enter")
#plot.distribution_comparison()
#plot.distributions()

#plot.print_statistics()
#plot.interval_type_counts()
#plot.interval_means()
#plot.interval_medians()
#plot.distribution_split()
#plot.interval_means_latex()
#plot.boxplot("filldir64-return:filldir64-enter")
#plot.boxplot4()
#plot.make_result_histogram()

#plot.distribution_split()
#plot.distribution_comparison()


#plot.interval_types_per_run()
#plot.interval_types_per_run2()

#plot.print_num_processes()

#plot.interval_types_per_run()
#plot.interval_types_per_run2()

#plot.print_num_events_per_process("x64_sys_call-enter")

#plot.plot_event_timeline()

