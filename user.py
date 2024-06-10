import os
import sys
from time import sleep

from data_classes import Event

from bcc import BPF
import threading
import subprocess

probe_points = [
    #"do_sys_openat2",
    #"x64_sys_call",  # maybe this only works as a retprobe: 'cannot attach kprobe, probe entry may not exist'
    "__x64_sys_getdents64",
    "__fdget_pos",
    "__fget_light",
    "iterate_dir",
    "security_file_permission",
    "apparmor_file_permission",
    "dcache_readdir",
    "filldir64",
    "verify_dirent_name",
    "touch_atime",
    "atime_needs_update",
]

stop = False
threads = []
programs = {}
output = []

for probe_point in probe_points:
    program_src = open("kernel.c").read()
    
    program_enter_src = program_src
    program_enter_src.replace("buffer", "buffer-" + probe_point + "-enter")
    program_enter_src.replace("12345", str(os.getpid()))
    bpf_enter_prog = BPF(text=program_enter_src)
    bpf_enter_prog.attach_kprobe(event=probe_point, fn_name="foo")
    programs[probe_point + "-enter"] = bpf_enter_prog

    def callback(ctx, data, size):
        global output
        event = bpf_enter_prog["buffer"].event(data)
        output.append(Event(probe_point, event.time, event.pid, event.tgid))

    bpf_enter_prog["buffer"].open_ring_buffer(callback)

    program_return_src = program_src
    program_return_src.replace("buffer", "buffer-" + probe_point + "-return")
    program_return_src.replace("12345", str(os.getpid()))
    bpf_return_prog = BPF(text=program_return_src)
    bpf_return_prog.attach_kretprobe(event=probe_point, fn_name="foo")
    programs[probe_point + "-return"] = bpf_return_prog

    def callback(ctx, data, size):
        global output
        event = bpf_return_prog["buffer"].event(data)
        output.append(Event(probe_point, event.time, event.pid, event.tgid))

    bpf_return_prog["buffer"].open_ring_buffer(callback)

print("BPF programs injected, buffering output...", file=sys.stderr)

finished = False
detection_PID = 0


def run_detection() -> None:
    sleep(2)
    global finished
    global detection_PID
    process = subprocess.Popen("./getpid_ls", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout_str = stdout.decode('utf-8')
    #print(stdout_str, file=sys.stderr)
    stderr_str = stderr.decode('utf-8')
    detection_PID = int((stdout_str.split("\n")[0])[4:])
    print("detection_PID: %i" % detection_PID, file=sys.stderr)
    sleep(5)
    finished = True


thread = threading.Thread(target=run_detection)
thread.start()

print(f"finished {finished}", file=sys.stderr)

while not finished:
    try:
        for probe_point, bpf_prog in programs.items():
            bpf_prog.ring_buffer_poll(30)  # TODO: think about the timeouts
    except KeyboardInterrupt:
        break

thread.join()

# detach all bpf probes
for bpf_prog in programs.values():
    bpf_prog.cleanup()

# It will be fun to write all the logged data somewhere, without creating syscalls...

print("Main loop exited, printing output.", file=sys.stderr)

output = sorted(output)  # sort by timestamp

output = [event for event in output if event.pid == detection_PID]

first = output[0].timestamp
for event in output:
    event.timestamp -= first

def print_output():
    print("probe_point\t\t\t\ttime\t\t\t\tpid\t\t\t\ttgid\n")
    for event in output:
        print("%s\t\t\t\t%lu\t\t\t\t%u\t\t\t\t%u" % (event.probe_point, event.timestamp, event.pid, event.tgid))
print_output()


import matplotlib.pyplot as plt

# Create the visualization
x = []  # Scatterplot X values
y = []  # Scatterplot Y Values

# Loop over the data a second time
for event in output:
    x.append(event.timestamp)
    y.append(event.probe_point)

plt.figure(figsize=(14,4))
plt.title("Timeline Plot")
plt.xlim(output[0].timestamp, output[-1].timestamp)
plt.scatter(x, y)

plt.savefig("timeline.svg")

exit(0)
