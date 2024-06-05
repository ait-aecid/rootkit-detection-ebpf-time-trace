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
    "verify_dirent_name"
]

stop = False
threads = []
programs = {}
output = []

for probe_point in probe_points:
    program_src = open("kernel.c").read()
    program_src.replace("buffer", "buffer-" + probe_point)
    program_src.replace("12345", str(os.getpid()))
    bpf_prog = BPF(text=program_src)
    bpf_prog.attach_kprobe(event=probe_point, fn_name="foo")
    programs[probe_point] = bpf_prog

    def callback(ctx, data, size):
        global output
        event = bpf_prog["buffer"].event(data)
        output.append(Event(probe_point, event.time, event.pid, event.tgid))

    bpf_prog["buffer"].open_ring_buffer(callback)

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
    print("collecting...", file=sys.stderr)
    try:
        print("collection loop head", file=sys.stderr)
        for probe_point, bpf_prog in programs.items():
            bpf_prog.ring_buffer_poll(30)  # TODO: think about the timeouts
        print("collected 1x", file=sys.stderr)
    except KeyboardInterrupt:
        break

thread.join()

# detach all bpf probes
for bpf_prog in programs.values():
    bpf_prog.cleanup()

# It will be fun to write all the logged data somewhere, without creating syscalls...

print("Main loop exited, printing output.", file=sys.stderr)

output = sorted(output)  # sort by timestamp

#output = [event for event in output if event.pid == detection_PID]

print("probe_point\ttime\tpid\ttgid\n")
for event in output:
    print("%s\t%lu\t%u\t%u\n" % (event.probe_point, event.timestamp, event.pid, event.tgid))

exit(0)
