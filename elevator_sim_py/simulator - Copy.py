import simpy
import pandas as pd

arrival_df_path = "low_dense_low_rise.csv"

num_elevators = 2
capacity = 15
travel_time_per_floor = 3
load_time_per_person = 1


# -----------------------------
# 乘客类
# -----------------------------
class Passenger:
    def __init__(self, env, pid, arrival_time, origin, destination, egcs, arrival_ts=None):
        self.env = env
        self.id = pid
        # arrival_time: numeric offset in seconds from simulation start (used by simpy)
        self.arrival_time = float(arrival_time)
        # arrival_ts: optional pandas.Timestamp representing the absolute wall-clock time
        self.arrival_ts = arrival_ts
        self.origin = origin
        self.destination = destination
        self.egcs = egcs
        # wait_start will be set when the passenger actually arrives (after the initial timeout)
        self.wait_start = None
        self.wait_time = None
        # aboard_time: env.now when boarded (numeric seconds)
        self.aboard_time = None
        # total_time: numeric duration from arrival until reaching destination
        self.total_time = None
        env.process(self.run())

    def run(self):
        # wait until the passenger's arrival time (numeric seconds)
        yield self.env.timeout(self.arrival_time)
        # mark the time they start waiting and request an elevator
        self.wait_start = self.env.now
        self.egcs.request_elevator(self)

# -----------------------------
# 电梯类
# -----------------------------
class Elevator:
    def __init__(self, env, eid, capacity=15, travel_time=3, load_time_per_person=1):
        self.env = env
        self.eid = eid
        self.capacity = capacity
        self.passengers = []
        self.floor = 1
        self.direction = 0  # 1 = up, -1 = down, 0 = idle
        self.requests = set()
        self.travel_time = travel_time
        self.load_time_per_person = load_time_per_person
        self.action = env.process(self.run())

    def move_to(self, floor):
        # 移动楼层
        distance = abs(self.floor - floor)
        if distance > 0:
            yield self.env.timeout(distance * self.travel_time)
            self.floor = floor

    def board_passengers(self, waiting_passengers):
        # 上客（考虑容量）
        available_space = self.capacity - len(self.passengers)
        if available_space <= 0:
            return []

        boarding = waiting_passengers[:available_space]
        load_time = len(boarding) * self.load_time_per_person
        yield self.env.timeout(load_time)
        self.passengers.extend(boarding)
        for p in boarding:
            # record when passenger boarded (numeric sim time) and their wait time
            p.aboard_time = self.env.now
            p.wait_time = self.env.now - p.wait_start

        return boarding

    def run(self):
        while True:
            if not self.requests:
                self.direction = 0
                yield self.env.timeout(1)
                continue

            target_floor = min(self.requests) if self.direction <= 0 else max(self.requests)
            self.direction = 1 if target_floor > self.floor else -1 if target_floor < self.floor else 0
            yield from self.move_to(target_floor)
            self.requests.discard(target_floor)
            yield self.env.timeout(1)  # 停留时间（开关门）

# -----------------------------
# 电梯群控系统 (EGCS)
# -----------------------------
class EGCS:
    def __init__(self, env, num_elevators=2, num_floors=10):
        self.env = env
        self.elevators = [Elevator(env, i+1, capacity, travel_time_per_floor, load_time_per_person) for i in range(num_elevators)]
        self.waiting = {i: [] for i in range(num_floors+1)}  # 每层的等待队列
        self.records = []
        # start_time will be a pandas.Timestamp representing the wall-clock time
        # corresponding to simulation time 0. Set by run_simulation after reading CSV.
        self.start_time = None

    def request_elevator(self, passenger):
        self.waiting[passenger.origin].append(passenger)
        elevator = self.assign_elevator(passenger)
        if elevator:
            self.env.process(self.handle_boarding(elevator, passenger.origin))

    def assign_elevator(self, passenger):
        origin = passenger.origin
        direction = 1 if passenger.destination > origin else -1

        # ① 优先选择同方向、靠近、且未满载的电梯
        candidates = [
            e for e in self.elevators
            if e.direction in (0, direction)
            and len(e.passengers) < e.capacity
        ]
        if candidates:
            return min(candidates, key=lambda e: abs(e.floor - origin))

        # ② 否则选择最近的空闲电梯
        idle = [e for e in self.elevators if e.direction == 0]
        if idle:
            return min(idle, key=lambda e: abs(e.floor - origin))
        return None

    def handle_boarding(self, elevator, floor):
        waiting = self.waiting[floor]
        if not waiting:
            return
        elevator.requests.add(floor)
        yield from elevator.move_to(floor)
        boarded = yield from elevator.board_passengers(waiting)
        self.waiting[floor] = waiting[len(boarded):]

        # 模拟电梯去目标层送人
        for p in boarded:
            yield from elevator.move_to(p.destination)
            # total time from arrival (wait_start) until reaching destination
            p.total_time = self.env.now - p.wait_start
            elevator.passengers.remove(p)
            # compute pandas Timestamp fields if start_time is available
            if self.start_time is not None:
                try:
                    arrival_ts = p.arrival_ts or (self.start_time + pd.Timedelta(seconds=p.arrival_time))
                except Exception:
                    arrival_ts = None
                board_ts = (self.start_time + pd.Timedelta(seconds=p.aboard_time)) if p.aboard_time is not None else None
                complete_ts = self.start_time + pd.Timedelta(seconds=self.env.now)
            else:
                arrival_ts = p.arrival_ts if hasattr(p, 'arrival_ts') else None
                board_ts = p.aboard_time
                complete_ts = self.env.now

            self.records.append({
                "PassengerID": p.id,
                "Origin": p.origin,
                "Destination": p.destination,
                "ArrivalTimestamp": arrival_ts,
                "BoardTimestamp": board_ts,
                "CompleteTimestamp": complete_ts,
                "WaitTime": p.wait_time,
                "TotalTime": p.total_time
            })

# -----------------------------
# 主函数
# -----------------------------
def run_simulation(csv_path, buffer_time=1000):
    env = simpy.Environment()


    # Read CSV and normalize/parse timestamps
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].notna().any():
            start_ts = df['timestamp'].min()
        else:
            start_ts = None
    else:
        start_ts = None

    num_floors = df[['origin', 'destination']].max().max()

    egcs = EGCS(env, num_elevators=num_elevators, num_floors=num_floors)
    # Store a wall-clock start time on EGCS for converting sim times to timestamps
    egcs.start_time = start_ts

    for idx, row in df.iterrows():
        if start_ts is not None and pd.notna(row['timestamp']):
            arrival_offset = (row['timestamp'] - start_ts).total_seconds()
            arrival_ts = row['timestamp']
        else:
            # fall back to a numeric offset field (timestamp may already be numeric), or 0
            arrival_offset = float(row.get('timestamp', row.get('time', 0)))
            arrival_ts = None

        pid = row.get('id', idx)
        Passenger(env, pid, arrival_offset, int(row['origin']), int(row['destination']), egcs, arrival_ts=arrival_ts)

    sim_time = df['timestamp'].max() - start_ts if start_ts is not None else df['timestamp'].max()
    sim_time = sim_time.total_seconds() + buffer_time
    
    env.run(until=sim_time)
    return pd.DataFrame(egcs.records).drop_duplicates()

# -----------------------------
# 示例运行
# -----------------------------
if __name__ == "__main__":
    df_result = run_simulation(arrival_df_path, buffer_time=600)
    df_result.to_csv(arrival_df_path.replace(".csv", "_simulation_results.csv"), index=False)
