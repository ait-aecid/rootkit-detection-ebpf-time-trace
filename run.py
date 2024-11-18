import sys
from process_data import Plot

plot = Plot(sys.argv)

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

