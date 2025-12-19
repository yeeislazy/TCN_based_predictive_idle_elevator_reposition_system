#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_tcn import TCN  # 来自 pytorch-tcn 库
from livelossplot import PlotLosses
import time


# In[2]:


num_epochs = 50
batch_size = 128


# In[3]:


class ElevatorCallsDataset(Dataset):
    def __init__(self, df, input_len=60*60, gap = 30 ,output_window=60):
        """
        df: pandas DataFrame with time series data (按时间排序，频次例如每秒／每分钟)
        input_len: 用多少时间步 (window length) 作为输入
        gap: 输入和输出之间的时间间隔（例如30表示预测输入和输出之间有30个秒的间隔）
        output_window: 预测多少步之后 (例如 60 表示预测下一分钟)
        feature_cols: list of feature列名 (包含楼层 call & direction one-hot + optional 时间特征)
        target_cols: list of target 列名 (未来是否有 call）
        """
        self.df = df.reset_index(drop=True)
        self.input_len = input_len
        self.gap = gap
        self.output_window = output_window

        data = df.values.astype(np.float32)
        self.samples = []

        total_length = len(data) - self.input_len - self.gap - self.output_window + 1
        total_length = max(0, total_length)

        for idx in range(total_length):
            x = data[idx:idx + self.input_len]
            output_window = data[idx + self.input_len + self.gap - 1: 
                                idx + self.input_len + self.gap + self.output_window - 1, 1:]
            y = (output_window.sum(axis=0) > 0).astype(np.float32)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)

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
        # 但 PyTorch-TCN 默认期望 (batch, channels, length)，因此需要转置
        x = x.transpose(1, 2)  # -> (batch, input_channels, seq_len)
        y = self.tcn(x)        # -> (batch, num_channels[-1], seq_len)
        # 取最后一个 time step’s feature map
        out = self.linear(y[:, :, -1])  # -> (batch, output_size)
        return out



# In[4]:


trainset = pd.read_csv('trainset.csv')  # 示例加载数据
trainset['timestamp'] = pd.to_datetime(trainset['timestamp'])

# timestamp to unix timestamp
trainset['timestamp'] = trainset['timestamp'].astype(np.int64) // 10**9
trainset.info()


# In[5]:


testset = pd.read_csv('testset.csv')
testset['timestamp'] = pd.to_datetime(testset['timestamp'])

    # timestamp to unix timestamp
testset['timestamp'] = testset['timestamp'].astype(np.int64) // 10**9


# In[6]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# In[7]:


train_dataset = ElevatorCallsDataset(trainset, input_len=60, gap=30, output_window=60)
train_loader = DataLoader(train_dataset, batch_size= batch_size , shuffle=True,pin_memory=True, num_workers =4)

test_dataset = ElevatorCallsDataset(testset, input_len=60, gap=30, output_window=60)
test_loader = DataLoader(test_dataset, batch_size= batch_size*2 , shuffle=False, num_workers =4)

model = ElevatorTCNModel(input_channels=len(trainset.columns), output_size=len(trainset.columns)-1)
model.to(device)

pos_count = torch.from_numpy(trainset.iloc[:, 1:].sum().to_numpy()).float()
neg_count = trainset.shape[0] - pos_count
pos_weight = neg_count / (pos_count + 1e-6)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

plot = PlotLosses()

start_time = time.time()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch).to(device)
            loss = criterion(logits, y_batch)
            test_loss += loss.item()

    epoch_loss /= len(train_loader)
    test_loss /= len(test_loader)
    scheduler.step()

    logs = {'loss': epoch_loss, 'val_loss': test_loss}
    plot.update(logs)
    plot.send()


end_time = time.time()
used_time = end_time - start_time
print(f"Training completed in {used_time} seconds")


# In[ ]:


# save the trained model
torch.save(model.state_dict(), 'elevator_tcn_model.pth')


# In[ ]:


import requests
import os
import time

headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjc1NTE1MywidXVpZCI6IjQwZDZlNjVhM2VmZWJlYjQiLCJpc19hZG1pbiI6ZmFsc2UsImJhY2tzdGFnZV9yb2xlIjoiIiwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.2oepLBMnnfPJVGzSGo2RI33NsnMmAZh1LCbL14AZEaMa55d55UM2C8CJioPiwoOjbfO8SQlZdVTiwky6GeljWg"}
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                     json={
                         "title": f"train_done, used time {used_time}, loss {test_loss}",
                         "name": "facial keypoint",
                         "content": "facial keypoint cnn training done"
                     }, headers = headers)
print(resp.content.decode())

time.sleep(30)

os.system("/usr/bin/shutdown")


# In[ ]:




