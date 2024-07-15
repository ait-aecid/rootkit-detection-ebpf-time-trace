from bcc import BPF
import os
import sys
import argparse
import threading
import subprocess
from time import sleep
from data_classes import Event, Experiment
from linux import shell, insert_rootkit, remove_rootkit, list_modules, ROOTKIT_NAME

probe_points = [
    #"do_sys_openat2",
    "x64_sys_call",  # maybe this only works as a retprobe: 'cannot attach kprobe, probe entry may not exist'
    "__x64_sys_getdents64",
    "__fdget_pos",
    #"__fget_light",
    "iterate_dir",
    #"security_file_permission",
    #"apparmor_file_permission",
    #"dcache_readdir",
    "filldir64",
    "verify_dirent_name",
    "touch_atime",
    #"atime_needs_update",
]

#stop = False
#threads = []
programs = {}
detection_PIDs = []
output = []

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", "-i", default=100, type=int, help="Number of times to run the experiment.")
parser.add_argument("--executable", "-e", default="./getpid_opendir_readdir_proc", type=str, help="Provide an executable for the experiment.")

args = parser.parse_args()

experiment = Experiment(args.executable, args.iterations, os.uname().release, [], [])


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
        #print("got data from " + probe_point + "-enter: " + str(event.time), file=sys.stderr)

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
        #print("got data from " + probe_point + "-return: " + str(event.time), file=sys.stderr)

    bpf_return_prog["buffer"].open_ring_buffer(callback)


def run_detection() -> None:
    global detection_PIDs
    process = subprocess.Popen(args.executable, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout_str = stdout.decode('utf-8')
    #print(stdout_str, file=sys.stderr)
    stderr_str = stderr.decode('utf-8')
    detection_PID = int((stdout_str.split("\n")[0])[4:])
    detection_PIDs.append(detection_PID)
    print("detection_PID: %i" % detection_PID, file=sys.stderr)
    process.wait()


def run_detection_Yx(Y: int):
    global finished
    for i in range(Y):
        print(f'Iteration {i}...', file=sys.stderr)
        run_detection()
    finished = True


for name, program in programs.items():
    print(name, file=sys.stderr)

print(f"Running experiment with {experiment.executable} for {experiment.iterations} times.", file=sys.stderr)

# make sure the rootkit is not loaded
if ROOTKIT_NAME in list_modules():
    print("rootkit was loaded! removing it...")
    remove_rootkit()

finished = False
thread = threading.Thread(target=run_detection_Yx, args=[args.iterations])
thread.start()

poll_count = 0
while not finished:
    for probe_point, bpf_prog in programs.items():
        poll_count += 1
        bpf_prog.ring_buffer_poll(5)
    sleep(0.0005)  # sleep for 5 milliseconds, then check the buffers again
print(f"polled {poll_count} times!", file=sys.stderr)

thread.join()

print("done with the \"no rootkit version\"", file=sys.stderr)

output = sorted(output)  # sort by timestamp

output = [event for event in output if event.pid in detection_PIDs]

# normalize time
first = output[0].timestamp
for event in output:
    event.timestamp -= first

# save events without rootkit
experiment.events = output



# --------------------------


# let's do the same thing again but with rootkit
print("loading rootkit...", file=sys.stderr)
insert_rootkit()

# reset global vars
output = []
finished = False

thread = threading.Thread(target=run_detection_Yx, args=[args.iterations])
thread.start()

print(f"finished: {finished}", file=sys.stderr)

poll_count = 0
while not finished:
    for probe_point, bpf_prog in programs.items():
        poll_count += 1
        bpf_prog.ring_buffer_poll(5)
    sleep(0.0005)  # sleep for 5 milliseconds, then check the buffers again
print(f"polled {poll_count} times!", file=sys.stderr)

thread.join()

print("done with the \"rootkit version\"", file=sys.stderr)

output = sorted(output)  # sort by timestamp

output = [event for event in output if event.pid in detection_PIDs]

# normalize time
first = output[0].timestamp
for event in output:
    event.timestamp -= first

# save events without rootkit
experiment.events_rootkit = output


# we are done

remove_rootkit()


# detach all bpf probes
for bpf_prog in programs.values():
    bpf_prog.cleanup()


print("Experiment finished, saving output.", file=sys.stderr)

import json
from datetime import datetime

filename = "output" + datetime.now().isoformat() + ".json"
with open(filename, 'w') as file:
    file.write(json.dumps(experiment, default=vars))

print(f"Saved data to %s" % filename)

# print size of saved file
shell("du -h " + filename)

exit(0)
