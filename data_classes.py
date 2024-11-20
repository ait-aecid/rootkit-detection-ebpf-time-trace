class Event:
    def __init__(self, probe_point: str, timestamp: int, pid: int, tgid: int):
        self.probe_point: str = probe_point
        self.timestamp: int = timestamp
        self.pid: int = pid
        self.tgid: int = tgid

    def __lt__(self, other: 'Event') -> bool:
        return self.timestamp < other.timestamp

    def __le__(self, other: 'Event') -> bool:
        return self.timestamp <= other.timestamp

    def __gt__(self, other: 'Event') -> bool:
        return self.timestamp > other.timestamp

    def __ge__(self, other: 'Event') -> bool:
        return self.timestamp >= other.timestamp

    def __getitem__(self, item):
        return self[item]


class Interval:
    def __init__(self, time: int, event_a: Event, event_b: Event, pid: int, tgid: int):
        self.time = time
        self.event_a = event_a
        self.event_b = event_b
        self.pid = pid
        self.tgid = tgid

    def __lt__(self, other: 'Interval') -> bool:
        return self.time < other.time

    def __le__(self, other: 'Interval') -> bool:
        return self.time <= other.time

    def __gt__(self, other: 'Interval') -> bool:
        return self.time > other.time

    def __ge__(self, other: 'Interval') -> bool:
        return self.time >= other.time

    def __getitem__(self, item):
        return self[item]


class Experiment:
    def __init__(self, executable: str, iterations: int, dir_content: str, linux_version: str, description: str, label: str = "", experiment_begin: str = None, experiment_end: str = None, events: [Event] = []):
        self.executable: str = executable
        self.iterations: int = iterations
        self.dir_content: str = dir_content
        self.linux_version: str = linux_version
        self.description: str = description
        self.label: str = label
        self.experiment_begin: str = experiment_begin
        self.experiment_end: str = experiment_end
        self.events: [Event] = events

    def __getitem__(self, item):
        return self[item]


def experiment_from_json(input: dict) -> Experiment:
    return Experiment(executable=input["executable"],
                      iterations=int(input["iterations"]),
                      dir_content=input["dir_content"],
                      linux_version=input["linux_version"],
                      description=input["description"],
                      experiment_begin=input["experiment_begin"],
                      experiment_end=input["experiment_end"],
                      events=[Event(**elem) for elem in input["events"]])
