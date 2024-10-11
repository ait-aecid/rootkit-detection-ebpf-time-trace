import sys
from process_data import Plot

plot = Plot(sys.argv)

#plot.sanity_check()

#plot.make_result_histogram()

#plot.get_slice_means([(0, 1650), (3550, 7000), (7000, 10300), (14840, 17440), (18270, 21470)], "filldir64-return:filldir64-enter")
plot.get_weighted_overal_slice_mean([(0, 1650), (3550, 7000), (7000, 10300), (14840, 17440), (18270, 21470)], "filldir64-return:filldir64-enter")

#plot.split("filldir64-return:filldir64-enter")
#plot.kmeans("filldir64-return:filldir64-enter")
#plot.distribution_comparison("filldir64-return:filldir64-enter")
#plot.distribution_comparison()
#plot.distributions()

#plot.print_statistics()
#plot.interval_type_counts()
plot.interval_means()
#plot.interval_medians()
#plot.distribution_split()
#plot.interval_means_latex()
#plot.boxplot("filldir64-return:filldir64-enter")
#plot.boxplot4()

#plot.distribution_split()
#plot.distribution_comparison()


#plot.interval_types_per_run()
#plot.interval_types_per_run2()

#plot.print_num_processes()

#plot.interval_types_per_run()
#plot.interval_types_per_run2()

#plot.print_num_events_per_process("x64_sys_call-enter")

#plot.plot_event_timeline()

