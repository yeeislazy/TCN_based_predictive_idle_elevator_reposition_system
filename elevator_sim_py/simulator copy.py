import simpy
import pandas as pd
import pygame
import threading
import time
import os
from collections import defaultdict

# =========================================================
# 模拟器逻辑部分（保持逻辑一致）
# =========================================================
class Passenger:
    def __init__(self, env, pid, arrival_time, origin, destination, egcs, arrival_ts=None):
        self.env = env
        self.id = pid
        self.arrival_time = float(arrival_time)
        self.arrival_ts = arrival_ts
        self.origin = origin
        self.destination = destination
        self.egcs = egcs
        self.wait_start_ts = None
        self.aboard_ts = None
        self.complete_ts = None
        self.waiting_time = None
        self.total_time = None
        env.process(self.run())

    def run(self):
        yield self.env.timeout(self.arrival_time)
        if getattr(self.egcs, 'start_time', None) is not None:
            self.wait_start_ts = self.egcs.start_time + pd.Timedelta(seconds=self.env.now)
        else:
            self.wait_start_ts = pd.Timestamp(0) + pd.Timedelta(seconds=self.env.now)
        self.egcs.request_elevator(self)


class Elevator:
    def __init__(self, env, eid, capacity=15, travel_time=3, load_time_per_person=1):
        self.env = env
        self.eid = eid
        self.capacity = capacity
        self.passengers = []
        self.floor = 1
        self.direction = 0
        self.requests = set()
        self.travel_time = travel_time
        self.load_time_per_person = load_time_per_person
        self.action = env.process(self.run())

    def move_to(self, floor):
        distance = abs(self.floor - floor)
        if distance > 0:
            travel_time = distance * self.travel_time
            steps = int(travel_time * 10)  # 更平滑的动画
            floor_step = (floor - self.floor) / steps
            for _ in range(steps):
                yield self.env.timeout(self.travel_time / steps)
                self.floor += floor_step
            self.floor = floor

    def board_passengers(self, waiting_passengers):
        available_space = self.capacity - len(self.passengers)
        if available_space <= 0:
            return []
        boarding = waiting_passengers[:available_space]
        load_time = len(boarding) * self.load_time_per_person
        yield self.env.timeout(load_time)
        self.passengers.extend(boarding)
        for p in boarding:
            if getattr(p.egcs, 'start_time', None):
                p.aboard_ts = p.egcs.start_time + pd.Timedelta(seconds=self.env.now)
            else:
                p.aboard_ts = pd.Timestamp(0) + pd.Timedelta(seconds=self.env.now)
            if p.wait_start_ts is not None:
                p.waiting_time = p.aboard_ts - p.wait_start_ts
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
            yield self.env.timeout(1)


class EGCS:
    def __init__(self, env, num_elevators=2, num_floors=10, capacity=15,
                 travel_time=3, load_time_per_person=1):
        self.env = env
        self.elevators = [Elevator(env, i+1, capacity, travel_time, load_time_per_person)
                          for i in range(num_elevators)]
        self.waiting = defaultdict(list)
        for f in range(1, num_floors + 1):
            self.waiting[f]
        self.records = []
        self.start_time = None

    def request_elevator(self, passenger):
        self.waiting[passenger.origin].append(passenger)
        elevator = self.assign_elevator(passenger)
        if elevator:
            self.env.process(self.handle_boarding(elevator, passenger.origin))

    def assign_elevator(self, passenger):
        origin = passenger.origin
        direction = 1 if passenger.destination > origin else -1
        candidates = [
            e for e in self.elevators
            if e.direction in (0, direction)
            and len(e.passengers) < e.capacity
        ]
        if candidates:
            return min(candidates, key=lambda e: abs(e.floor - origin))
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
        for p in boarded:
            yield from elevator.move_to(p.destination)
            if getattr(self, 'start_time', None) is not None:
                p.complete_ts = self.start_time + pd.Timedelta(seconds=self.env.now)
            else:
                p.complete_ts = pd.Timestamp(0) + pd.Timedelta(seconds=self.env.now)
            if p.wait_start_ts is not None:
                p.total_time = p.complete_ts - p.wait_start_ts
            elevator.passengers.remove(p)
            self.records.append({
                "PassengerID": p.id,
                "Origin": p.origin,
                "Destination": p.destination,
                "ArrivalTimestamp": p.arrival_ts,
                "BoardTimestamp": p.aboard_ts,
                "CompleteTimestamp": p.complete_ts,
                "WaitingDuration": p.waiting_time,
                "TotalDuration": p.total_time
            })


# =========================================================
# 实时模拟控制逻辑
# =========================================================
def setup_simulation(params):
    env = simpy.Environment()
    df = pd.read_csv(params['arrival_df_path'])
    num_floors = int(params['num_floors'])
    egcs = EGCS(env,
                num_elevators=int(params['num_elevators']),
                num_floors=num_floors,
                capacity=int(params['capacity']),
                travel_time=float(params['travel_time_per_floor']),
                load_time_per_person=float(params['load_time_per_person']))

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        start_ts = df['timestamp'].min() if df['timestamp'].notna().any() else None
    else:
        start_ts = None
    egcs.start_time = start_ts

    for idx, row in df.iterrows():
        if start_ts is not None and pd.notna(row.get('timestamp')):
            arrival_offset = (row['timestamp'] - start_ts).total_seconds()
            arrival_ts = row['timestamp']
        else:
            arrival_offset = float(row.get('arrival_time', 0))
            arrival_ts = None
        pid = row.get('id', idx)
        Passenger(env, pid, arrival_offset, int(row['origin']), int(row['destination']), egcs, arrival_ts=arrival_ts)
    return env, egcs


# =========================================================
# PyGame 实时动画显示
# =========================================================
pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Elevator Simulation (Realtime)")

font = pygame.font.Font(None, 28)
clock = pygame.time.Clock()

params = {
    "arrival_df_path": "low_dense_low_rise.csv",
    "num_elevators": "2",
    "capacity": "15",
    "travel_time_per_floor": "3",
    "load_time_per_person": "1",
    "num_floors": "10",
}

egcs_instance = None
env_instance = None
simulation_running = False
status_text = "Idle"

def draw_ui():
    screen.fill((245, 245, 245))
    base_y = 550
    floor_height = 40
    if egcs_instance:
        colors = [(255,0,0),(0,128,255),(0,200,0),(255,165,0)]
        for i, e in enumerate(egcs_instance.elevators):
            y = base_y - e.floor * floor_height
            pygame.draw.rect(screen, colors[i % len(colors)], (100 + i*70, y-30, 50, 30))
            pygame.draw.rect(screen, (0,0,0), (100 + i*70, y-30, 50, 30), 2)
            label = font.render(f"E{i+1}  F:{e.floor:.1f}", True, (0,0,0))
            screen.blit(label, (100 + i*70, y-55))
        for i in range(1, int(params["num_floors"])+1):
            y = base_y - i*floor_height
            pygame.draw.line(screen, (180,180,180), (80, y), (400, y))
            fl_text = font.render(str(i), True, (0,0,0))
            screen.blit(fl_text, (50, y-10))

    pygame.draw.rect(screen, (100,200,100), (700, 500, 140, 50))
    screen.blit(font.render("Start", True, (255,255,255)), (740, 515))
    screen.blit(font.render(status_text, True, (0,0,0)), (100, 500))


def main_loop():
    global egcs_instance, env_instance, simulation_running, status_text
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.Rect(700,500,140,50).collidepoint(event.pos):
                    if not simulation_running:
                        env_instance, egcs_instance = setup_simulation(params)
                        simulation_running = True
                        status_text = "Running..."

        if simulation_running and env_instance:
            try:
                env_instance.step()
            except Exception:
                simulation_running = False
                status_text = "✅ Simulation complete"

        draw_ui()
        pygame.display.flip()
        clock.tick(30)
        time.sleep(0.1)  # 控制仿真速度

    pygame.quit()


if __name__ == "__main__":
    main_loop()
