from bcc import BPF
import os
import sys
from datetime import datetime
import argparse
import threading
import subprocess
import random
import string
import gzip
from time import sleep
from data_classes import Event, Experiment
from linux import shell, insert_rootkit, remove_rootkit, list_modules, ROOTKIT_NAME, run_background

MAGIC_STRING = "caraxes"

probe_points = [
    #"do_sys_openat2",
    #"x64_sys_call",  # maybe this only works as a retprobe: 'cannot attach kprobe, probe entry may not exist' <- Linux < 6.5
    #"__x64_sys_getdents64",
    #"__fdget_pos",
    #"__fget_light",
    "iterate_dir",
    #"security_file_permission",
    #"apparmor_file_permission",
    "dcache_readdir",
    "filldir64",
    "verify_dirent_name",
    "touch_atime",
    #"atime_needs_update",
    #"__f_unlock_pos"
]

programs = {}
detection_PIDs = []
output = []

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", "-i", default=100, type=int, help="Number of times to run the experiment.")
parser.add_argument("--executable", "-e", default="ls", type=str, help="Provide an executable for the experiment.")
parser.add_argument("--normal", "-n", action='store_true', help="Run the normal execution, without anomalies.")
parser.add_argument("--rootkit", "--anomalous", "-r", "-a", action='store_true', help="Run the anomalous execution, with rootkit.")
parser.add_argument("--drop-boundary-events", "-d", action='store_true', help="Drop all events of the first and last PID of each run.\nThese events often miss data.\nMay lead to empty output file if runs <= 2.")
parser.add_argument("--load", "-l", default="", nargs="?", const="stress-ng --cpu 10", help="Put the system under load during the experiment. You can provide a custom executable to do so. Default is 'stress-ng --cpu 10'. Consider shell escaping.")
parser.add_argument("--hidden-files", default=1, type=int, help="Specify the number of hidden files to create.")
parser.add_argument("--visible-files", default=1, type=int, help="Specify the number of visible files to create.")
parser.add_argument("--file-name-length", default=8, type=int, help="Specify the general length of names of files that will be created. Too short will behave bad. MAGIC_WORD for hidden file gets added.")
parser.add_argument("--description", default="", type=str, help="Description of the current experiment, this will be saved in the output's metadata.")

args = parser.parse_args()

if args.normal is False and args.rootkit is False:
    # By default, run experiment with and without rootkit
    args.normal = True
    args.rootkit = True

# setup directory structure
DIR_NAME = 'test_dir_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

shell("mkdir " + DIR_NAME)  # in the CWD is fine

for i in range(args.visible_files):
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=args.file_name_length))
    shell("touch " + DIR_NAME + "/" + name)

for i in range(args.hidden_files):
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=args.file_name_length))
    index = int(random.random() * (args.file_name_length+1))
    name = name[0:index] + "_" + MAGIC_STRING + "_" + name[index:args.file_name_length]
    shell("touch " + DIR_NAME + "/" + name)

dir_content = shell("ls -a1 " + DIR_NAME, mute=True).replace("\n", ",")


experiment = Experiment(executable=args.executable,
                        iterations=args.iterations,
                        dir_content=dir_content,
                        linux_version=os.uname().release,
                        drop_boundary_events=args.drop_boundary_events,
                        load=args.load,
                        hidden_files=args.hidden_files,
                        visible_files=args.visible_files,
                        description=args.description)

print("compiling eBPF probes...", file=sys.stderr)
for probe_point in probe_points:
    program_src = open("kernel.c").read()

    program_enter_src = program_src
    program_enter_src.replace("buffer", "buffer-" + probe_point + "-enter")
    program_enter_src.replace("12345", str(os.getpid()))
    bpf_enter_prog = BPF(text=program_enter_src, cflags=["-Wno-macro-redefined"])
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
    bpf_return_prog = BPF(text=program_return_src, cflags=["-Wno-macro-redefined"])
    bpf_return_prog.attach_kretprobe(event=probe_point, fn_name="foo")
    programs[probe_point + "-return"] = bpf_return_prog

    def callback(ctx, data, size):
        global output
        event = bpf_return_prog["buffer"].event(data)
        output.append(Event(probe_point, event.time, event.pid, event.tgid))
        #print("got data from " + probe_point + "-return: " + str(event.time), file=sys.stderr)

    bpf_return_prog["buffer"].open_ring_buffer(callback)
print("probes compiled!", file=sys.stderr)

def run_detection_once(error_on_hidden: bool) -> None:
    global detection_PIDs
    process = subprocess.Popen([args.executable, DIR_NAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout_str = stdout.decode('utf-8')
    if (MAGIC_STRING in stdout_str) != error_on_hidden:
        raise Exception("rootkit failed! error_on_hidden: " + str(error_on_hidden) + "; ls result: <" + stdout_str.replace('\n','+') + ">")
    detection_PIDs.append(process.pid)
    print("detection_PID: %i" % process.pid, file=sys.stderr)
    process.wait()


def run_detection(iterations: int, error_on_hidden: bool):
    global finished
    for i in range(iterations):
        print(f'Iteration {i}...', file=sys.stderr)
        run_detection_once(error_on_hidden)
    finished = True


print("\nattached bpf probes:", file=sys.stderr)
for name, program in programs.items():
    print(name, file=sys.stderr)
print("")  # newline

print(f"Running experiment with {experiment.executable} for {experiment.iterations} times.", file=sys.stderr)

experiment.experiment_begin = datetime.now().isoformat()

normal_events = []
if args.normal:
    # make sure the rootkit is not loaded
    if ROOTKIT_NAME in list_modules():
        print("rootkit was loaded! removing it...")
        remove_rootkit()

    if args.load:
        stress_process = run_background(args.load)

    finished = False
    detection_thread = threading.Thread(target=run_detection, args=[args.iterations, True])
    detection_thread.start()

    poll_count = 0
    while not finished:
        for probe_point, bpf_prog in programs.items():
            poll_count += 1
            bpf_prog.ring_buffer_poll(5)
        sleep(0.0005)  # sleep for 5 milliseconds, then check the buffers again
    print(f"polled {poll_count} times!", file=sys.stderr)

    detection_thread.join()
    if args.load:
        stress_process.kill()

    print("done with the \"no rootkit version\"", file=sys.stderr)

    output = sorted(output)  # sort by timestamp

    output = [event for event in output if event.pid in detection_PIDs]

    # normalize time
    first = output[0].timestamp
    for event in output:
        event.timestamp -= first

    # save events without rootkit
    normal_events = output



# --------------------------

rootkit_events = []
if args.rootkit:
    # let's do the same thing again but with rootkit

    if not ROOTKIT_NAME in list_modules():
        print("loading rootkit...", file=sys.stderr)
        insert_rootkit()
    else:
        print("rootkit was already loaded! that's weird...")

    # reset global vars
    output = []
    finished = False

    if args.load:
        stress_process = run_background(args.load)

    detection_thread = threading.Thread(target=run_detection, args=[args.iterations, False])
    detection_thread.start()

    print(f"finished: {finished}", file=sys.stderr)

    poll_count = 0
    while not finished:
        for probe_point, bpf_prog in programs.items():
            poll_count += 1
            bpf_prog.ring_buffer_poll(5)
        sleep(0.0005)  # sleep for 5 milliseconds, then check the buffers again
    print(f"polled {poll_count} times!", file=sys.stderr)

    detection_thread.join()
    if args.load:
        stress_process.kill()

    print("done with the \"rootkit version\"", file=sys.stderr)

    output = sorted(output)  # sort by timestamp

    output = [event for event in output if event.pid in detection_PIDs]

    # normalize time
    first = output[0].timestamp
    for event in output:
        event.timestamp -= first

    # save events with rootkit
    rootkit_events = output

    # we are done
    remove_rootkit()

experiment.experiment_end = datetime.now().isoformat()

# detach all bpf probes
for bpf_prog in programs.values():
    bpf_prog.cleanup()

# cleanup testdir structure
shell("rm -rf " + DIR_NAME)

# delete first and last PID
if args.drop_boundary_events:
    # normal
    lowest_PID = min(normal_events, key=lambda event: event.pid)
    highest_PID = max(normal_events, key=lambda event: event.pid)
    lowest_PID = lowest_PID.pid
    highest_PID = highest_PID.pid
    print("lowest: " + str(lowest_PID) + "    highest: " + str(highest_PID), file=sys.stderr)
    normal_events = [event for event in normal_events if event.pid != lowest_PID]
    normal_events = [event for event in normal_events if event.pid != highest_PID]

    # normal
    lowest_PID = min(rootkit_events, key=lambda event: event.pid)
    highest_PID = max(rootkit_events, key=lambda event: event.pid)
    lowest_PID = lowest_PID.pid
    highest_PID = highest_PID.pid
    print("lowest: " + str(lowest_PID) + "    highest: " + str(highest_PID), file=sys.stderr)
    rootkit_events = [event for event in rootkit_events if event.pid != lowest_PID]
    rootkit_events = [event for event in rootkit_events if event.pid != highest_PID]

print("Experiment finished, saving output.", file=sys.stderr)

if not os.path.exists("events"):
    os.makedirs("events")

import json

if args.normal:
    filename = "events/events_" + datetime.now().isoformat() + "_normal.json.gz"
    experiment.events = normal_events
    experiment.label = "normal"
    with gzip.open(filename, 'w', compresslevel=1) as file:
        file.write(json.dumps(experiment, default=vars).encode())

if args.rootkit:
    filename = "events/events_" + datetime.now().isoformat() + "_rootkit.json.gz"
    experiment.events = rootkit_events
    experiment.label = "rootkit"
    with gzip.open(filename, 'w', compresslevel=1) as file:
            file.write(json.dumps(experiment, default=vars).encode())

print(f"Saved data to %s" % filename)

# print size of saved file
shell("du -h " + filename)

exit(0)
