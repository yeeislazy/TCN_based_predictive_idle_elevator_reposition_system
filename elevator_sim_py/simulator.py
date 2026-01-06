import simpy
import pandas as pd
import numpy as np
import pygame
import time
import tkinter as tk
import torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_tcn import TCN
import os

mode = 'call_record'  # 'animation' or 'simulate' or 'call_record'
reposition_mode = None  # 'tcn' or 'tsai' or 'none'
num_floors=20

sim_speed = 1.0  # simulation speed multiplier

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
arrival_df_path = os.path.join(parent_dir, 'new_arrival_simulator', 'data', 'low_dense_low_rise.csv')
output_records_path = arrival_df_path.replace(".csv", "_simulation_records.csv")

tcn_path = '../elevator-tcn/best_model/best_precision_modelmodel_training-v5_restart_focalloss_alpha_0.25_33.pth'

num_elevators = 2
capacity = 15
travel_time_per_floor = 3
load_time_per_person = 1

class ElevatorTCNModel(nn.Module):
    def __init__(self, input_channels, output_size, num_channels=[64, 64, 64], kernel_size=3, dropout=0.1):
        super().__init__()
        self.tcn = TCN(num_inputs=input_channels,
                       num_channels=num_channels,
                       kernel_size=kernel_size,
                       dropout=dropout,
                       causal=True)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_channels)  
        # 但 PyTorch-TCN 默认期望 (batch, channels, length),因此需要转置
        x = x.transpose(1, 2)  # -> (batch, input_channels, seq_len)
        y = self.tcn(x)        # -> (batch, num_channels[-1], seq_len)
        # 取最后一个 time step’s feature map
        out = self.linear(y[:, :, -1])  # -> (batch, output_size)
        return out

class ElevatorCallsDataset(Dataset):
    def __init__(self, df, input_len=60*60, gap = 30 ,output_window=60,downsample_seconds = 60):
        """
        df: pandas DataFrame with time series data (按时间排序,频次例如每秒／每分钟)
        input_len: 用多少时间步 (window length) 作为输入
        gap: 输入和输出之间的时间间隔（例如30表示预测输入和输出之间有30个秒的间隔）
        output_window: 预测多少步之后 (例如 60 表示预测下一分钟)
        feature_cols: list of feature列名 (包含楼层 call & direction one-hot + optional 时间特征)
        target_cols: list of target 列名 (未来是否有 call）
        """
        self.df = df.reset_index(drop=True)
        self.data = self.df.values
        self.input_len = input_len
        self.gap = gap
        self.output_window = output_window

        self.downsample_seconds = downsample_seconds

        self.total_length = len(self.data) - input_len - gap - output_window + 1
        self.total_length = max(self.total_length, 0)
            
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        input_window = self.data[idx:idx + self.input_len]
    
        x = []
        for i in range(0, self.input_len, self.downsample_seconds):
            block = input_window[i : i + self.downsample_seconds]
            x.append(block.sum(axis=0))
    
        x = np.stack(x).astype(np.float32)
    
        output_window = self.data[
            idx + self.input_len + self.gap - 1:
            idx + self.input_len + self.gap + self.output_window - 1, 3:]
        
        y = (output_window.sum(axis=0) > 0).astype(np.float32)
    
        return torch.from_numpy(x), torch.from_numpy(y)

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
    def __init__(self, env, num_elevators=2, num_floors=10, idle_reposition=False,idle_reposition_coldown_duration=180):
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
        self.baseline_wait_time = 15.19  # baseline without idle repositioning
        self.awt = 0
        self.reposition_count = 0
        self.model_triggered_count = 0
        
        self.idle_time = 0
        self.idle_reposition_wait = 10
        
        self.idle_reposition_coldown = 0
        self.idle_reposition_coldown_duration = idle_reposition_coldown_duration
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #import machine learning model for repositioning
        if reposition_mode == 'tcn':
            input_channels=num_floors*2+3
            output_size=num_floors*2
            self.model = ElevatorTCNModel(input_channels=input_channels, output_size=output_size)
            self.model.load_state_dict(torch.load(tcn_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
        

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
                                        
            if vacancy and self.idle_reposition and mode != 'call_record':
                if self.idle_time < self.idle_reposition_wait:
                    self.idle_time += 1
                else:
                    self.idle_time = 0
                    if self.idle_reposition_coldown > 0:
                        self.idle_reposition_coldown -= 1
                    else:
                        self.idle_reposition_coldown = self.idle_reposition_coldown_duration
                        self.idle_repositioning()
            
            # 每隔一定时间扫描一次
            yield self.env.timeout(1)  # 可以根据仿真需求改成 0.5 或 2 秒
            
    def add_to_waiting(self, passenger):
        direction = 1 if passenger.destination > passenger.origin else -1
        self.waiting[passenger.origin][direction].append(passenger)
        passenger.assigned = None
        
    def model_input(self):
        max_time = self.start_time + pd.Timedelta(seconds=self.env.now)
        min_time = max_time - pd.Timedelta(hours=1)
        
        columns = [ str(floor) + direction for floor in range(self.num_floors) for direction in ['_Up', '_Down']]
        rows = pd.date_range(start=min_time, end=max_time, freq="S")
        
        call_record = pd.DataFrame(0, index=rows, columns = ['timestamp','day','time']+columns)
        call_record['timestamp'] = call_record.index
        call_record['day'] = call_record.index.weekday
        call_record['time'] = call_record.index.hour * 3600 + call_record.index.minute * 60 + call_record.index.second
        
        
        def preprocess_calls(row):
            if row.isna().any():
                return
            time = row['Timestamp']
            floor = row['Floor']
            direction = 'Up' if row['Direction'] == 1 else 'Down'
        
            call_record.at[time, f"{floor}_{direction}"] = 1        
        
        call_records_df = pd.DataFrame(self.call_records)
        # only record within last 1 hour
        call_records_df = call_records_df[(call_records_df['Timestamp'] >= min_time) & (call_records_df['Timestamp'] <= max_time)]
        call_records_df.apply(preprocess_calls, axis=1)
        
        call_record = call_record.drop(columns=['0_Down', f'{self.num_floors-1}_Up'])
        call_record['timestamp'] = call_record['timestamp'].astype(np.int64) // 10**9
        
        input_window = call_record.values
        x = []
        for i in range(0, 60*60, 60):
            block = input_window[i : i + 60]
            x.append(block.sum(axis=0))
    
        x = np.stack(x).astype(np.float32)
        return torch.from_numpy(x)
        

    def idle_repositioning(self):
        if reposition_mode == 'tcn':
            x = self.model_input().to(self.device)
            self.model_triggered_count += 1
            logits = self.model(x.unsqueeze(0))
            logits = torch.clamp(logits, -20, 20)
            preds = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
            
            if preds.sum() == 0:
                return
            
            standby_indices = [i for i, v in enumerate(preds) if v >= 0.3]
            standby_floors = set(i // 2 for i in standby_indices)
            print(len(standby_floors))
            
        if reposition_mode == 'none':
            current_time = self.start_time + pd.Timedelta(seconds=self.env.now)
            if current_time.hour >= 7 and current_time.hour < 10:
                standby_floors = [0]
            else:
                return

        for standby_floor in standby_floors:
            idle_elevators = [e for e in self.elevators if e.direction == 0 and not e.requests]
            if idle_elevators:
                for e in idle_elevators:
                    if e.floor != standby_floor:
                        e.requests.add(standby_floor)
                        self.reposition_count += 1
                        e.direction = -1 if e.floor > standby_floor else 1
                        
                        
    def average_wait_time(self):
        if not self.records:
            return 0
        total_wait = sum(r['WaitTime'] for r in self.records)
        awt =  total_wait / len(self.records)
        self.awt = awt
        return awt

# -----------------------------
# 主函数
# -----------------------------
def setup_simulation(csv_path, buffer_time=1000):
    env = simpy.Environment()


    # Read CSV and normalize/parse timestamps
    df = pd.read_csv(csv_path)
    if mode != 'call_record':
        index = len(df) * 0.8
        df = df.iloc[int(index):]
    
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].notna().any():
            start_ts = df['timestamp'].min()
        else:
            start_ts = None
    else:
        start_ts = None

    # num_floors = df[['origin', 'destination']].max().max()

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

def simulate_loop(arrival_df_path):
    env, egcs, min_timestamp, sim_time = setup_simulation(arrival_df_path, buffer_time=600)
    if reposition_mode:
        egcs.idle_reposition = True
    env.run(until=sim_time)    
    # Save records to CSV
    if mode == 'simulate':
        records_df = pd.DataFrame(egcs.records)
        records_df.to_csv(output_records_path, index=False)
        awt = egcs.average_wait_time()
        print(f'Prediction model triggered count: {egcs.model_triggered_count}')
        print(f"Reposition count: {egcs.reposition_count}")
        print(f"Simulation complete. Average Wait Time: {awt}s")
        return awt
    
    if mode == 'call_record':
        call_records_df = pd.DataFrame(egcs.call_records)
        call_records_path = arrival_df_path.replace(".csv", "_call_records.csv")
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

def animation_loop(awt,arrival_df_path):
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

        # draw control buttons and info
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
        
        # total completed passengers
        tcp = 0
        if egcs_instance and egcs_instance.records:
            tcp = len(egcs_instance.records)
        screen.blit(font.render(f"Completed Passengers: {tcp}", True, (0, 0, 0)), (WIDTH*0.65-15, 230))
        
        # average passenger per day
        apd = 0
        if egcs_instance and egcs_instance.records and min_time:
            days_passed = (min_time + pd.Timedelta(seconds=env_instance.now) - min_time).days + 1
            apd = tcp / days_passed
        screen.blit(font.render(f"Avg Passengers/Day: {apd:.1f}", True, (0, 0, 0)), (WIDTH*0.65-15, 260))
        
        # baseline average waiting time
        bawt = 15.19
        screen.blit(font.render(f"Baseline AWT: {bawt:.2f}s", True, (0, 0, 0)), (WIDTH*0.65-15, 300))
        
        
        # average waiting time
        awt = 0
        if egcs_instance and egcs_instance.records:
            awt = egcs_instance.average_wait_time()
        screen.blit(font.render(f"Current AWT: {awt:.2f}s", True, (0, 0, 0)), (WIDTH*0.65-15, 330))

        
        # reposition count
        rc = 0
        if egcs_instance:
            rc = egcs_instance.reposition_count
        screen.blit(font.render(f"Reposition Count: {rc}", True, (0, 0, 0)), (WIDTH*0.65-15, 370))
        
        # reposition count per day
        rcpd = 0
        if egcs_instance and min_time:
            days_passed = (min_time + pd.Timedelta(seconds=env_instance.now) - min_time).days + 1
            rcpd = rc / days_passed
        screen.blit(font.render(f"Reposition Count/Day: {rcpd:.1f}", True, (0, 0, 0)), (WIDTH*0.65-15, 400))


    # ==== 主循环 ====
    def main_loop(awt,arrival_df_path):
        global egcs_instance, env_instance, simulation_running, status_text, elevators, sim_speed
        env_instance, egcs_instance, min_timestamp, _ = setup_simulation(arrival_df_path, buffer_time=600)
        draw_ui(progress='init',min_time=min_timestamp)
        egcs_instance.baseline_wait_time = awt
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
    
    main_loop(awt,arrival_df_path)


if __name__ == "__main__":
    if mode == 'animation' :
        awt = simulate_loop(arrival_df_path)
        animation_loop(awt,arrival_df_path)
    elif mode == 'simulate' :
        simulate_loop(arrival_df_path)
    elif mode == 'call_record' :    
        for file in ["low_dense_low_rise.csv", "low_dense_high_rise.csv", "high_dense_low_rise.csv", "high_dense_high_rise.csv"]:
            arrival_df_path = os.path.join(parent_dir, "new_arrival_simulator", "data", file)
            simulate_loop(arrival_df_path)



