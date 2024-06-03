import os
import sys

from bcc import BPF
import threading

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
        output += (probe_point, event.time, event.pid, event.tgid)

    bpf_prog["buffer"].open_ring_buffer(callback)

print("BPF programs injected, buffering output...", file=sys.stderr)

while True:
    try:
        for probe_point, bpf_prog in programs.items():
            bpf_prog.ring_buffer_poll(30)  # TODO: think about the timeouts
    except KeyboardInterrupt:
        break

# detach all bpf probes
for bpf_prog in programs.values():
    bpf_prog.cleanup()

# It will be fun to write all the logged data somewhere, without creating syscalls...

print("Main loop exited, printing output.", file=sys.stderr)

output_sorted = sorted(output, key=lambda x: x[1])  # sort by 2nd key in tuple (timestamp)

print("probe_point\ttime\tpid\ttgid\n")
for probe_point, time, pid, tgid in output_sorted:
    print("%s\t%lu\t%u\t%u\n" % (probe_point, time, pid, tgid))

exit(0)
