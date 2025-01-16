cd /sys/kernel/debug/tracing # navigate to the tracer
echo function_graph > current_tracer # select the wanted tracer
echo 1 > tracing_on; # turn the tracing on
ls ; tree # trigger getdents
echo 0 > tracing_on; # turn tracing off
less trace # inspect the output
