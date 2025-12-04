import simpy
import pandas as pd
import pygame
import time
import tkinter as tk

mode = 'simulate'  # 'animation' or 'simulate'

sim_speed = 1.0  # simulation speed multiplier


arrival_df_path = r"E:\iCloudDrive\master of applied computing\capstone project\new_arrival_simulator\low_dense_low_rise.csv"
output_records_path = arrival_df_path.replace(".csv", "_simulation_records.csv")


num_elevators = 2
capacity = 15
travel_time_per_floor = 3
load_time_per_person = 1



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
        self.egcs.add_to_waiting(self)

# -----------------------------
# 电梯类
# -----------------------------
class Elevator:
    def __init__(self, env, egcs, eid, capacity=15, travel_time=3, load_time_per_person=1):
        self.env = env
        self.egcs = egcs
        self.eid = eid
        self.capacity = capacity
        self.passengers = []
        self.floor = 0
        self.destination = None
        self.direction = 0  # 1 = up, -1 = down, 0 = idle
        self.moving = 0
        self.requests = set()
        self.travel_time = travel_time
        self.load_time_per_person = load_time_per_person
        self.action = env.process(self.run())

    def move_to(self, floor):
        # 移动楼层
        self.destination = floor
        self.moving = 1 if floor > self.floor else -1 if floor < self.floor else 0
        step = 1 / self.travel_time

        while abs(self.floor - floor) > step / 2:
            yield self.env.timeout(1)
            self.floor += self.moving * step

            if (self.moving > 0 and self.floor > floor) or (self.moving < 0 and self.floor < floor):
                self.floor = floor
        
        self.floor = floor
        self.moving = 0

    def board_passengers(self, waiting_passengers):
        # 上客（考虑容量）
        available_space = self.capacity - len(self.passengers)
        if available_space <= 0:
            return []

        boarding = waiting_passengers[:available_space]
        if waiting_passengers:
            self.moving = False
            self.passengers.extend(boarding)
            for p in boarding:
                yield self.env.timeout(self.load_time_per_person)
                # record when passenger boarded (numeric sim time) and their wait time
                p.aboard_time = self.env.now
                p.wait_time = self.env.now - p.wait_start

            return boarding
        else:
            return []

    def run(self):
        while True:
            # 1️⃣ 没有任务则等 1s
            if not self.requests:
                self.moving = False
                yield self.env.timeout(1)
                continue

            # 2️⃣ 取下一个任务楼层
            if self.direction == 1:
                next_floor = min(self.requests)
            else:
                next_floor = max(self.requests)

            # move to next floor
            yield from self.move_to(next_floor)

            self.moving = False
            disembarked = [p for p in self.passengers if p.destination == next_floor]
            for p in disembarked:
                yield self.env.timeout(self.load_time_per_person)
                p.total_time = self.env.now - p.wait_start  # 统计总耗时
                self.passengers.remove(p)
                self.egcs.records.append({
                    "PassengerID": p.id,
                    "Origin": p.origin,
                    "Destination": p.destination,
                    "ArrivalTimestamp": p.arrival_ts,
                    "BoardTimestamp": (self.egcs.start_time + pd.Timedelta(seconds=p.aboard_time)) if self.egcs.start_time is not None and p.aboard_time is not None else p.aboard_time, 
                    "CompleteTimestamp": (self.egcs.start_time + pd.Timedelta(seconds=self.env.now)) if self.egcs.start_time is not None else self.env.now,
                    "WaitTime": p.wait_time,
                    "TotalTime": p.total_time
                })


            # 3️⃣ 上客
            # if self.requests:
            #     # 如果电梯空闲，决定当前方向
            #     next_floor = min(self.requests, key=best_key)
            #     self.direction = 1 if next_floor > self.floor else -1
            # else:
            #     self.direction = 0

            waiting = self.egcs.waiting[next_floor][self.direction]
            if waiting:
                boarded = yield from self.board_passengers(waiting)
                self.egcs.waiting[next_floor][self.direction] = waiting[len(boarded):]
                
                for p in self.egcs.waiting[next_floor][self.direction]:
                    # mark that these passengers are still waiting (not boarded yet)
                    p.assigned = None
                # Mark floor as no longer reserved
                self.egcs.floor_reserved[next_floor][self.direction] = None
                
                if next_floor in self.requests:
                    self.requests.discard(next_floor)
                
                # 4️⃣ 送乘客到目的地
                for p in boarded:
                    if p.destination not in self.requests:
                        self.requests.add(p.destination)
            else:
                self.requests.discard(next_floor)
            
            if not self.requests:
                self.direction = 0
# -----------------------------
# 电梯群控系统 (EGCS)
# -----------------------------
class EGCS:
    def __init__(self, env, num_elevators=2, num_floors=10, idle_reposition=False):
        self.env = env
        self.elevators = [Elevator(env, self, i+1, capacity, travel_time_per_floor, load_time_per_person) for i in range(num_elevators)]
        self.num_floors = num_floors + 1
        self.waiting = {i: {1:[], -1:[]} for i in range(self.num_floors)}  # 每层的等待队列
        self.floor_reserved = {i: {1: None, -1: None} for i in range(self.num_floors)}  # 每层的预留电梯
        self.records = []
        self.call_records = []
        # start_time will be a pandas.Timestamp representing the wall-clock time
        # corresponding to simulation time 0. Set by run_simulation after reading CSV.
        self.start_time = None
        self.idle_reposition = idle_reposition
        self.env.process(self.dispatcher())

    # def request_elevator(self, passenger):
    #     # 把乘客放入等待队列
    #     if passenger.destination > passenger.origin:
    #         self.waiting[passenger.origin][1].append(passenger)  # 上行队列
    #     else:
    #         self.waiting[passenger.origin][-1].append(passenger)  # 下行队列
    #     # 分配电梯（返回 Elevator 实例或 None）
    #     elevator = self.assign_elevator(passenger)
    #     if elevator:
    #         # 把乘客所在楼层加入该电梯的 requests（pickup）
    #         elevator.requests.add(passenger.origin)
    #         # 如果电梯空闲，设置方向预期（便于 assign 以后仍然匹配）
    #         if elevator.direction == 0:
    #             elevator.direction = 1 if passenger.destination > passenger.origin else -1
    #         # debug
    #         # print(f"[t={self.env.now}] Assigned elevator {elevator.eid} to pickup at {passenger.origin} for pid {passenger.id}")
    #     else:
    #         # no elevator currently assignable -> it stays in waiting; later idle elevator will scan waiting
    #         # print(f"[t={self.env.now}] No elevator assigned yet for pickup at {passenger.origin} pid {passenger.id}")
    #         pass


    def assign_elevator(self, direction, origin):
        # origin = passenger.origin
        # direction = 1 if passenger.destination > origin else -1

        # prefer elevators already moving in that direction, and not full
        candidates = []
        for e in self.elevators:
            if e.direction in (0, direction) and len(e.passengers) < e.capacity:
                if e.moving == 1 and e.floor > origin:
                    continue  # 向上但已超过该楼层
                if e.moving == -1 and e.floor < origin:
                    continue
                candidates.append(e)
                
        if candidates:
            e = min(candidates, key=lambda e: abs(e.floor - origin))
            e.direction = direction
            return e

        # ② 否则选择最近的空闲电梯
        idle = [e for e in self.elevators if e.direction == 0]
        if idle:
            e = min(idle, key=lambda e: abs(e.floor - origin))
            e.direction = direction
            return e
        return None
    
    def dispatcher(self):
        while True:
            # 遍历每层楼的等待队列
            vacancy = True
            for floor, queues in self.waiting.items():
                for direction in [1, -1]:
                    if self.floor_reserved[floor][direction] is None:
                        waiting_list = queues[direction]
                        if not waiting_list:
                            continue
                        
                        if waiting_list:
                            self.call_records.append({
                                "Timestamp": (self.start_time + pd.Timedelta(seconds=self.env.now)) if self.start_time is not None else self.env.now,
                                "Floor": floor,
                                "Direction": direction})
                        
                        vacancy = False
                        elevator = self.assign_elevator(direction, floor)
                        if elevator:
                            self.floor_reserved[floor][direction] = elevator
                            elevator.requests.add(floor)
                            unassigned = [p for p in waiting_list if not p.assigned]
                            for passenger in unassigned:
                                passenger.assigned = elevator  # 标记已分配
                                # 如果电梯空闲，设方向
                                if elevator.direction == 0:
                                        elevator.direction = direction
                                        
            if vacancy and self.idle_reposition:
                self.idle_repositioning()
            
            # 每隔一定时间扫描一次
            yield self.env.timeout(1)  # 可以根据仿真需求改成 0.5 或 2 秒
            
    def add_to_waiting(self, passenger):
        direction = 1 if passenger.destination > passenger.origin else -1
        self.waiting[passenger.origin][direction].append(passenger)
        passenger.assigned = None

    def idle_repositioning(self):
        current_time = self.start_time + pd.Timedelta(seconds=self.env.now)
        if current_time.hour >= 7 and current_time.hour < 10:
            standby_floor = 0

            idle_elevators = [e for e in self.elevators if e.direction == 0 and not e.requests]
            if idle_elevators:
                avg_floor = sum(e.floor for e in self.elevators) / len(self.elevators)
                for e in idle_elevators:
                    if e.floor != standby_floor:
                        e.requests.add(standby_floor)
                        e.direction = -1 if e.floor > standby_floor else 1

# -----------------------------
# 主函数
# -----------------------------
def setup_simulation(csv_path, buffer_time=1000):
    env = simpy.Environment()


    # Read CSV and normalize/parse timestamps
    origin_df = pd.read_csv(csv_path)
    index = int(len(origin_df)*0.8)
    while origin_df["time_second"].iloc[index] > 370*60:
        index -= 1
        df = origin_df.iloc[index:]
    
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
    min_timestamp = df["timestamp"].min()
    

    return env, egcs, min_timestamp, sim_time

def simulate_loop():
    env, egcs, min_timestamp, sim_time = setup_simulation(arrival_df_path, buffer_time=600)
    env.run(until=sim_time)
    records_df = pd.DataFrame(egcs.records)
    records_df.to_csv(output_records_path, index=False)
    
    call_records_df = pd.DataFrame(egcs.call_records)
    call_records_path = output_records_path.replace('.csv', '_calls.csv')
    call_records_df.to_csv(call_records_path, index=False)



egcs_instance = None
env_instance = None
simulation_running = False
status_text = "Idle"
elevators = []

# =========================================================
# PyGame 实时动画显示
# =========================================================
import pygame, time

def animation_loop():
    global egcs_instance, env_instance, simulation_running, status_text, elevators
    
    pygame.init()
    WIDTH, HEIGHT = 750, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Elevator Simulation (Realtime)")

    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    # ==== 模拟参数 ====
    params = {
        "arrival_df_path": "low_dense_low_rise.csv",
        "num_elevators": "2",
        "capacity": "15",
        "travel_time_per_floor": "3",
        "load_time_per_person": "1",
        "num_floors": "10",
    }

    # ==== 电梯动画类 ====
    class ElevatorAnimation:
        def __init__(self,index, base_y, passenger_count=0, floor_height=50):
            self.index = index
            self.x = 100 + index * 70
            self.y = base_y + int(floor_height * 0.05)
            self.base_y = base_y
            self.elevator_height = floor_height - int(floor_height * 0.1)
            self.image = pygame.Surface((40, self.elevator_height))
            self.color = (100, 100, 255)
            self.rect = self.image.get_rect(topleft=(self.x, self.y))
            self.passenger_count_text = None
            self.floor_height = floor_height
            self.update()

        def update(self):
            elevator = egcs_instance.elevators[self.index]
            
            self.y = self.base_y  - (elevator.floor+1) * self.floor_height + self.floor_height * 0.05
            self.y = int(self.y)
            self.rect.y = self.y
            self.image.fill(self.color)
            self.passenger_count_text = font.render(str(len(elevator.passengers)), True, (255, 255, 255))
            self.image.blit(self.passenger_count_text, (5, 5))

        def draw(self, screen):
            screen.blit(self.image, self.rect)


    # ==== UI 绘制函数 ====
    def draw_ui(progress='update',simulation_running = False, min_time = None):
        global status_text
        screen.fill((245, 245, 245))
        base_y = 650
        floor_height = (base_y - 10) // (int(egcs_instance.num_floors) if egcs_instance else int(params["num_floors"]) + 1)
        y_elevator = []

        # 绘制楼层
        if egcs_instance:
            for i in range(egcs_instance.num_floors):
                y = base_y - i * floor_height         
                pygame.draw.line(screen, (180, 180, 180), (80, y), (400, y))
                if i == 0:
                    fl_text = font.render("G", True, (0, 0, 0))
                else:
                    fl_text = font.render(str(i), True, (0, 0, 0))
                screen.blit(fl_text, (50, y - 30))
                num_waiting = sum(len(v) for v in egcs_instance.waiting[i].values())
                num_waiting_text = font.render(str(num_waiting), True, (0, 0, 0))
                screen.blit(num_waiting_text, (410, y - 30))

        # 初始化电梯动画
        if progress == 'init':
            elevators.clear()
            for i, e in enumerate(egcs_instance.elevators):
                elevators.append(ElevatorAnimation(i, base_y, floor_height=floor_height))
                elevators[-1].draw(screen)
                clock_text = min_time.strftime('%Y-%m-%d %H:%M:%S')
                day = min_time.weekday()
                status_text = "Paused"

        # 更新电梯动画
        elif progress == 'update':
            for i, e in enumerate(egcs_instance.elevators):
                elevators[i].update()
                elevators[i].draw(screen)
                if min_time:
                    current_time = (min_time + pd.Timedelta(seconds=env_instance.now))
                else:
                    current_time = pd.Timestamp.now()
                clock_text = current_time.strftime('%Y-%m-%d %H:%M:%S')
                day = current_time.weekday()

        # 绘制 Start 按钮与状态
        if simulation_running:
            button_color = (200, 100, 100)
            button_text = "Pause"
        else:
            button_color = (100, 200, 100)
            button_text = "Start"
        pygame.draw.rect(screen, button_color, (WIDTH*0.75-50, 500, 140, 50))
        
        if egcs_instance.idle_reposition:
            idle_button_color = (100, 200, 100)
            idle_button_text = "Reposition: ON"
        else:
            idle_button_color = (200, 100, 100)
            idle_button_text = "Reposition: OFF"
        pygame.draw.rect(screen, idle_button_color, (WIDTH*0.75-65, 560, 180, 50))
        screen.blit(font.render(idle_button_text, True, (255, 255, 255)), (WIDTH*0.75-53 , 575))
        
        screen.blit(font.render(button_text, True, (255, 255, 255)), (WIDTH*0.75-5 , 515))
        screen.blit(font.render(status_text, True, (0, 0, 0)), (100, base_y + 20))
        screen.blit(font.render(clock_text, True, (0, 0, 0)), (WIDTH*0.65-15, 50))
        screen.blit(font.render(f"Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day]}", True, (0, 0, 0)), (WIDTH*0.65-15, 80))
        screen.blit(font.render(f"Speed: ", True, (0, 0, 0)), (WIDTH*0.65-15, 110))
        screen.blit(font.render(f"{sim_speed}x", True, (200, 100, 100)), (WIDTH*0.65+55, 110))
        screen.blit(font.render(f"[1] 0.5x [2] 1x    [3] 5x", True, (0, 0, 0)), (WIDTH*0.65-15, 140))
        screen.blit(font.render(f"[4] 10x  [5] 50x [6] 100x", True, (0, 0, 0)), (WIDTH*0.65-15, 170))
        
        # average waiting time
        awt = 0
        if egcs_instance and egcs_instance.records:
            awt = sum(r['WaitTime'] for r in egcs_instance.records) / len(egcs_instance.records)
        screen.blit(font.render(f"Avg Wait Time: {awt:.2f}s", True, (0, 0, 0)), (WIDTH*0.65-15, 230))
        
        # total completed passengers
        tcp = 0
        if egcs_instance and egcs_instance.records:
            tcp = len(egcs_instance.records)
        screen.blit(font.render(f"Completed Passengers: {tcp}", True, (0, 0, 0)), (WIDTH*0.65-15, 260))
        
        # average passenger per day
        apd = 0
        if egcs_instance and egcs_instance.records and min_time:
            days_passed = (min_time + pd.Timedelta(seconds=env_instance.now) - min_time).days + 1
            apd = tcp / days_passed
        screen.blit(font.render(f"Avg Passengers/Day: {apd:.1f}", True, (0, 0, 0)), (WIDTH*0.65-15, 290))


    # ==== 主循环 ====
    def main_loop():
        global egcs_instance, env_instance, simulation_running, status_text, elevators, sim_speed
        env_instance, egcs_instance, min_timestamp, _ = setup_simulation(arrival_df_path, buffer_time=600)
        draw_ui(progress='init',min_time=min_timestamp)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.Rect(WIDTH*0.75-40, 500, 140, 50).collidepoint(event.pos):
                        simulation_running = not simulation_running
                        status_text = "Running..." if simulation_running else "Paused"
                        
                    if pygame.Rect(WIDTH*0.75-60, 560, 180, 50).collidepoint(event.pos):
                        egcs_instance.idle_reposition = not egcs_instance.idle_reposition
                        status_text = "Idle Repositioning ON" if egcs_instance.idle_reposition else "Idle Repositioning OFF"
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # 空格暂停/继续
                        simulation_running = not simulation_running
                        status_text = "Running..." if simulation_running else "Paused"
                    elif event.key == pygame.K_1: sim_speed = 0.5
                    elif event.key == pygame.K_2: sim_speed = 1
                    elif event.key == pygame.K_3: sim_speed = 5
                    elif event.key == pygame.K_4: sim_speed = 10
                    elif event.key == pygame.K_5: sim_speed = 50
                    elif event.key == pygame.K_6: sim_speed = 100
                
                
            if simulation_running:                                           
                try:
                    target_time = env_instance.now + sim_speed
                    while env_instance.now < target_time:
                        env_instance.step()

                except Exception:
                    simulation_running = False
                    status_text = "✅ Simulation complete"
                    
            draw_ui(progress='update', simulation_running=simulation_running, min_time=min_timestamp)
            pygame.display.flip()
            clock.tick(10)

        pygame.quit()
    
    main_loop()


if __name__ == "__main__":
    if mode == 'animation':
        animation_loop()
    elif mode == 'simulate':
        simulate_loop()


