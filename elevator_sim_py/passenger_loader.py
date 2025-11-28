import csv
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np


@dataclass
class Passenger:
    """Represents a passenger in the elevator simulation.

    Fields:
      id: optional passenger id (string)
      arrival_time: simulation time when passenger arrives at origin floor
      origin_floor: integer origin floor
      des_floor: integer destination floor
      aboard_time: time when passenger boards the elevator (None until boarded)
      state: 'waiting' | 'traveling' | 'arrived'
      complete_time: time when passenger reaches destination (None until arrived)
      waiting_time: computed as aboard_time - arrival_time once boarded (None until boarded)
    """

    arrival_time: float
    origin_floor: int
    des_floor: int
    aboard_time: Optional[float] = None
    state: str = field(default='waiting')
    complete_time: Optional[float] = None
    waiting_time: Optional[float] = None

    def board(self, time: float) -> None:
        """Mark the passenger as boarded at simulation time `time`.
        Sets aboard_time, computes waiting_time, and updates state to 'traveling'."""
        if self.aboard_time is None and self.state == 'waiting':
            self.aboad_time_set(time)

    def abooad_time_set(self, time: float) -> None:
        # small internal helper to avoid repeated checks in unit tests
        self.aboad_time = float(time)
        self.waiting_time = self.aboad_time - float(self.arrival_time)
        self.state = 'traveling'

    def arrive(self, time: float) -> None:
        """Mark the passenger as arrived at their destination at simulation time `time`."""
        self.complete_time = float(time)
        self.state = 'arrived'

    def to_dict(self) -> dict:
        """Return a plain dict representation (backwards compatible)."""
        return {
            'id': self.id,
            'time': self.arrival_time,
            'origin': self.origin_floor,
            'destination': self.des_floor,
            'aboard_time': self.aboad_time if hasattr(self, 'aboad_time') else self.aboad_time,
            'state': self.state,
            'complete_time': self.complete_time,
            'waiting_time': self.waiting_time,
        }


def load_passengers_from_csv(path: str) -> List[Passenger]:
    """Load passengers from a CSV file and return a list of Passenger objects.

    Expected CSV columns (header optional):
      time, origin, destination, id
    """
    passengers: List[Passenger] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        # If file doesn't have header, fallback to simple reader
        if reader.fieldnames is None:
            f.seek(0)
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                time, origin, destination = row[:3]
                pid = row[3] if len(row) > 3 else None
                passengers.append(Passenger(
                    id=pid or f"p{len(passengers)+1}",
                    arrival_time=float(time),
                    origin_floor=int(origin),
                    des_floor=int(destination),
                ))
        else:
            # Normalize header names
            for row in reader:
                if not any(row.values()):
                    continue

                def g(k, default=None):
                    for h in row:
                        if h.strip().lower() == k:
                            return row[h]
                    return default

                t = g('time')
                o = g('origin') or g('floor')
                d = g('destination') or g('dest')
                pid = g('id') or g('passenger')
                if t is None or o is None or d is None:
                    # skip malformed row
                    continue
                passengers.append(Passenger(
                    id=pid or f"p{len(passengers)+1}",
                    arrival_time=float(t),
                    origin_floor=int(float(o)),
                    des_floor=int(float(d)),
                ))
    # sort by arrival time
    passengers.sort(key=lambda p: p.arrival_time)
    return passengers
