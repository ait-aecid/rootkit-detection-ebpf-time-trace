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
