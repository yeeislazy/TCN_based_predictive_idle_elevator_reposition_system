import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_tcn import TCN  # 来自 pytorch-tcn 库
from livelossplot import PlotLosses
import time
from livelossplot.outputs import MatplotlibPlot
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, precision_score,precision_recall_curve





num_epochs = 50
batch_size = 128
weights = [0,0.25,0.5,0.75,1]

train_dir = "trainset.csv"
test_dir = "testset.csv"

current_file_dir = Path(__file__).parent.resolve()
print(current_file_dir)
model_dir = current_file_dir / 'models_v4'
img_dir = current_file_dir / 'imgs_v4'
board_dir = current_file_dir / "runs"
model_dir.mkdir(parents=True, exist_ok=True)
img_dir.mkdir(parents=True, exist_ok=True)
board_dir.mkdir(parents=True, exist_ok=True)



model_df = pd.DataFrame(columns=['Model','weight', 'epoch','acc', 'precision', 'recall'])




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
        self.input_len = input_len
        self.gap = gap
        self.output_window = output_window

        data = df.values.astype(np.float32)
        self.samples = []

        total_length = len(data) - self.input_len - self.gap - self.output_window + 1
        total_length = max(0, total_length)

        for idx in range(total_length):
            input_window = data[idx:idx + self.input_len]
            x = []
            for i in range(0, input_len, downsample_seconds):
                block = input_window[i : i + downsample_seconds]
                x.append(block.sum(axis=0)) 
            x = np.stack(x)

            output_window = data[idx + self.input_len + self.gap - 1: 
                                idx + self.input_len + self.gap + self.output_window - 1, 3:]
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
        # 但 PyTorch-TCN 默认期望 (batch, channels, length),因此需要转置
        x = x.transpose(1, 2)  # -> (batch, input_channels, seq_len)
        y = self.tcn(x)        # -> (batch, num_channels[-1], seq_len)
        # 取最后一个 time step’s feature map
        out = self.linear(y[:, :, -1])  # -> (batch, output_size)
        return out



# clear previous run
for filename in model_dir.iterdir():
    file_path = model_dir / filename
    if file_path.is_file():
        file_path.unlink()
for filename in img_dir.iterdir():
    file_path = img_dir / filename
    if file_path.is_file():
        file_path.unlink()


trainset = pd.read_csv(train_dir)  # 示例加载数据
trainset['timestamp'] = pd.to_datetime(trainset['timestamp'])

# timestamp to unix timestamp
trainset['timestamp'] = trainset['timestamp'].astype(np.int64) // 10**9
trainset.info()




testset = pd.read_csv(test_dir)
testset['timestamp'] = pd.to_datetime(testset['timestamp'])

    # timestamp to unix timestamp
testset['timestamp'] = testset['timestamp'].astype(np.int64) // 10**9




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')




# class LossSmoother:
#     def __init__(self, alpha=0.9, clip_value=10):
#         self.alpha = alpha
#         self.clip_value = clip_value
#         self.smoothed = None

#     def update(self, value):
#         # 裁剪异常值
#         value = min(value, self.clip_value)

#         if self.smoothed is None:
#             self.smoothed = value
#         else:
#             self.smoothed = self.alpha * self.smoothed + (1 - self.alpha) * value
#         return self.smoothed



train_dataset = ElevatorCallsDataset(trainset, input_len=60*60, gap=30, output_window=60)
train_loader = DataLoader(train_dataset, batch_size= batch_size , shuffle=True,pin_memory=True, num_workers =4)

test_dataset = ElevatorCallsDataset(testset, input_len=60*60, gap=30, output_window=60)
test_loader = DataLoader(test_dataset, batch_size= batch_size*2 , shuffle=False, num_workers =4)

num_labels = len(trainset.columns) - 3

total_counts = torch.zeros(num_labels)
positive_counts = torch.zeros(num_labels)

with torch.no_grad():
    for x_batch, y_batch in train_loader:
        # y_batch shape: (batch, output_window=60, num_labels)
        # 展平成 (batch*60, num_labels)
        y_flat = y_batch.reshape(-1, num_labels)

        total_counts += y_flat.shape[0]
        positive_counts += y_flat.sum(dim=0)

negative_counts = total_counts - positive_counts
positive_rate = positive_counts / total_counts

origin_pos_weight = (negative_counts / (positive_counts + 1e-5))





def search_best_thresholds(all_probs, all_labels):
    num_labels = all_probs.shape[1]
    best_thresholds = np.zeros(num_labels)

    for i in range(num_labels):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]

        if y_true.sum() == 0:
            best_thresholds[i] = 0.5
            continue

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        idx = f1.argmax()
        if idx >= len(thresholds):
            best_thresholds[i] = 0.5
        else:
            best_thresholds[i] = thresholds[idx]

    return best_thresholds

def evaluate_model(all_probs,all_labels, threshold):
    num_labels = all_probs.shape[1]
    accuracies = []
    precisions = []
    recalls = []

    for i in range(num_labels):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]
        y_pred = (y_prob >= threshold[i]).astype(np.float32)

        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

    return np.mean(accuracies), np.mean(precisions), np.mean(recalls)





for weight in weights:
    print(f"weight:{weight}")
    model = ElevatorTCNModel(input_channels=len(trainset.columns), output_size=len(trainset.columns)-3)
    model.to(device)
    
    pos_weight = origin_pos_weight.clone()
    for i in range(pos_weight.shape[0]):
        if pos_weight[i] > 1:
            pos_weight[i] = 1 + (pos_weight[i] - 1)*weight
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    warmup_epochs = int(num_epochs * 0.4)    
    cosine = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    
    log_dir = f"restart_thresholdsearch_weight{weight}"
    log_dir = board_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)  
    writer = SummaryWriter(log_dir)
    
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_recalls = []
    val_recalls = []
    train_precisions = []
    val_precisions = []
    
    thresholds = np.full(num_labels,0.5)
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        train_accuracy = 0.0
    
        # 用于计算recall的统计量
        total_true_positives = 0
        total_actual_positives = 0
        total_predicted_positives = 0
    
        all_probs = []
        all_labels = []
    
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            logits = torch.clamp(logits, -20, 20)
            loss = criterion(logits, y_batch)
            train_loss += loss.item()
    
            probs = torch.sigmoid(logits)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            all_probs.append(probs.detach())
            all_labels.append(y_batch)
    
        if epoch > warmup_epochs:
            adjusted_epoch = epoch - warmup_epochs
            cosine.step(adjusted_epoch)
    
        all_probs = torch.cat(all_probs).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
    
        thresholds = search_best_thresholds(all_probs, all_labels)
    
        train_accuracy, train_precision, train_recall = evaluate_model(all_probs, all_labels, thresholds)    
    
    
    
        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            test_accuracy = 0.0
    
            test_recall = 0.0
    
            test_true_positives = 0
            test_actual_positives = 0
            test_predicted_positives = 0
    
            all_probs = []
            all_labels = []    
    
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch).to(device)
                logits = torch.clamp(logits, -20, 20)
                loss = criterion(logits, y_batch)
                test_loss += loss.item()
    
                probs = torch.sigmoid(logits)
    
                all_probs.append(probs.detach())
                all_labels.append(y_batch)
    
        all_probs = torch.cat(all_probs).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
    
        test_accuracy, val_precision, val_recall = evaluate_model(all_probs, all_labels, thresholds)
    
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", test_loss, epoch)
        writer.add_scalar("Acc/train", train_accuracy, epoch)
        writer.add_scalar("Acc/val", test_accuracy, epoch)
        writer.add_scalar("Recall/train", train_recall, epoch)
        writer.add_scalar("Recall/val", val_recall, epoch)
        writer.add_scalar("Precision/train", train_precision, epoch)
        writer.add_scalar("Precision/val", val_precision, epoch)
    
        train_losses.append(train_loss)
        val_losses.append(test_loss)
        train_accs.append(train_accuracy)
        val_accs.append(test_accuracy)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
    
        model_name = f"weight_{weight}_epoch{epoch:02d}_acc{test_accuracy:.4f}_prec{val_precision:.4f}_rec{val_recall:.4f}.pth"
        torch.save(model.state_dict(), model_dir / model_name)
        model_df.loc[len(model_df)] = {
        'Model': model_name,
        'weight': weight,
        'epoch': epoch,
        'acc': test_accuracy,
        'precision': val_precision,
        'recall': val_recall
    }
        for i, ths in enumerate(thresholds):
            model_df.loc[len(model_df)-1, f'threshold_label_{i}'] = ths
    
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ---- Loss ----
    axes[0, 0].plot(train_losses, label="Train Loss")
    axes[0, 0].plot(val_losses, label="Validation Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    
    # ---- Accuracy ----
    axes[0, 1].plot(train_accs, label="Train Accuracy")
    axes[0, 1].plot(val_accs, label="Validation Accuracy")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    
    # ---- Precision ----
    axes[1, 0].plot(train_precisions, label="Train Precision")
    axes[1, 0].plot(val_precisions, label="Validation Precision")
    axes[1, 0].set_title("Precision")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")
    
    # ---- Recall ----
    axes[1, 1].plot(train_recalls, label="Train Recall")
    axes[1, 1].plot(val_recalls, label="Validation Recall")
    axes[1, 1].set_title("Recall")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    
    plt.suptitle(f"Training Metrics (weight={weight})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    img_name = f"metrics_weight_{weight}.png"
    plt.savefig(img_dir / img_name , dpi=300, bbox_inches="tight")
    plt.close()
    
    
    end_time = time.time()
    used_time = end_time - start_time
    print(f"Training completed in {used_time} seconds")
    
    writer.close()






# save the trained model
torch.save(model.state_dict(), 'elevator_tcn_model_final.pth')

model_df.to_csv(model_dir / 'model_df.csv', index=False)





import requests
import os
import time

headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjc1NTE1MywidXVpZCI6IjQwZDZlNjVhM2VmZWJlYjQiLCJpc19hZG1pbiI6ZmFsc2UsImJhY2tzdGFnZV9yb2xlIjoiIiwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.2oepLBMnnfPJVGzSGo2RI33NsnMmAZh1LCbL14AZEaMa55d55UM2C8CJioPiwoOjbfO8SQlZdVTiwky6GeljWg"}
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                     json={
                         "title": f"train_done, used time {used_time}, loss {test_loss}",
                         "name": f"train_done, used time {used_time}, loss {test_loss}",
                         "content": "facial keypoint cnn training done"
                     }, headers = headers)
print(resp.content.decode())

time.sleep(90)







# os.system("/usr/bin/shutdown")

