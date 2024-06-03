BPF_RINGBUF_OUTPUT(buffer, 1 << 4);  // TODO: does this buffer size make sense?

struct event {
    unsigned long time;
    u32 pid;
    u32 tgid;
};

int foo(struct pt_regs *ctx) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = (u32) pid_tgid;
    bpf_trace_printk("pid is: %ui", pid);

    /** Exclude the PID of the userspace tracing program.
        This number gets replaced before BPF compilation.
        Since the tracing program also does the compilation before
        before the probe gets inserted, the PID is known ahead.
    */
    if(pid == 12345){
        return 0;
    }

    struct event *event = buffer.ringbuf_reserve(sizeof(struct event));
    if (!event) {
        return 1;
    }

    event->tgid = pid_tgid >> 32;
    event->pid = (u32) pid_tgid;
    event->time = bpf_ktime_get_ns();

    buffer.ringbuf_submit(event, 0);
    // or, to discard: buffer.ringbuf_discard(event, 0);

    return 0;
}