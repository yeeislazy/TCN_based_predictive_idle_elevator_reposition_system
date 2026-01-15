# %%
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
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter
from Pathlib import Path

num_epochs = 50
batch_size = 128
weights = [0,0.25,0.5,0.75,1]

train_dir = "low_dense_low_rise_trainset.csv"
test_dir = "low_dense_low_rise_testset.csv"

current_file_dir = Path(__file__).parent.resolve()
print(current_file_dir)
model_dir = current_file_dir / 'models_v2'
img_dir = current_file_dir / 'imgs_v2'
board_dir = current_file_dir / "runs"

model_df = pd.DataFrame(columns=['Model','weight', 'epoch','acc', 'precision', 'recall'])

class ElevatorCallsDataset(Dataset):
    def __init__(self, df, input_len=60*60, gap = 30 ,output_window=60,downsample_seconds = 60):
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
        x = x.transpose(1, 2)  # -> (batch, input_channels, seq_len)
        y = self.tcn(x)        # -> (batch, num_channels[-1], seq_len)
        # 取最后一个 time step’s feature map
        out = self.linear(y[:, :, -1])  # -> (batch, output_size)
        return out



trainset = pd.read_csv(train_dir).astype(np.float32)
testset = pd.read_csv(test_dir).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# clear previous run
for filename in model_dir.iterdir():
    file_path = model_dir / filename
    if file_path.is_file():
        file_path.unlink()
for filename in img_dir.iterdir():
    file_path = img_dir / filename
    if file_path.is_file():
        file_path.unlink()

train_dataset = ElevatorCallsDataset(trainset, input_len=60*60, gap=30, output_window=5*60)
train_loader = DataLoader(train_dataset, batch_size= batch_size , shuffle=True,pin_memory=True, num_workers =4)
print('train loader done')

test_dataset = ElevatorCallsDataset(testset, input_len=60*60, gap=30, output_window=5*60)
test_loader = DataLoader(test_dataset, batch_size= batch_size*2 , shuffle=False, num_workers =4)
print('test loader done')

num_labels = len(trainset.columns) - 3

total_counts = torch.zeros(num_labels)
positive_counts = torch.zeros(num_labels)

with torch.no_grad():
    for x_batch, y_batch in train_loader:
        # y_batch shape: (batch, output_window=60, num_labels)
        # exp (batch*60, num_labels)
        y_flat = y_batch.reshape(-1, num_labels)

        total_counts += y_flat.shape[0]
        positive_counts += y_flat.sum(dim=0)

negative_counts = total_counts - positive_counts
positive_rate = positive_counts / total_counts

origin_pos_weight = (negative_counts / (positive_counts + 1e-5))
print('Start Training')
total_start_time = time.time()
for weight in weights:
    start_time = time.time()
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
    
    warmup = LinearLR(optimizer, start_factor=0.7, end_factor=1.0, total_iters=20)
    
    cosine = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[20])

    log_dir = f"v2_restart_noweightclamp_{weight}"
    log_dir = os.path.join(board_dir,log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_recalls = []
    val_recalls = []
    train_precisions = []
    val_precisions = []


    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        model.train()
        epoch_accuracy = 0.0
        
        # for recall and precision
        total_true_positives = 0
        total_actual_positives = 0
        total_predicted_positives = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            logits = torch.clamp(logits, -20, 20)
            loss = criterion(logits, y_batch)
            epoch_loss += loss.item()

            
            preds = (torch.sigmoid(logits) > 0.5).float()
            epoch_accuracy += (preds == y_batch).float().mean().item()

            # calculate recall
            true_positives = ((preds == 1) & (y_batch == 1)).float().sum().item()
            actual_positives = (y_batch == 1).float().sum().item()
            
            total_true_positives += true_positives
            total_actual_positives += actual_positives
            
            # calculate precision
            predicted_positives = (preds == 1).float().sum().item()
            
            total_predicted_positives += predicted_positives
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_recall = total_true_positives / (total_actual_positives + 1e-5)
        train_precision = total_true_positives / (total_predicted_positives + 1e-5)
            
        
        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            test_accuracy = 0.0

            test_recall = 0.0

            test_true_positives = 0
            test_actual_positives = 0
            test_predicted_positives = 0
            
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch).to(device)
                logits = torch.clamp(logits, -20, 20)
                loss = criterion(logits, y_batch)
                test_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                test_accuracy += (preds == y_batch).float().mean().item()

                # calculate validation recall
                batch_true_positives = ((preds == 1) & (y_batch == 1)).float().sum().item()
                batch_actual_positives = (y_batch == 1).float().sum().item()
                
                #
                batch_predicted_positives = (preds == 1).float().sum().item()
                
                test_true_positives += batch_true_positives
                test_actual_positives += batch_actual_positives
                test_predicted_positives += batch_predicted_positives

        val_recall = test_true_positives / (test_actual_positives + 1e-5)
        val_precision = test_true_positives / (test_predicted_positives + 1e-5)
            
        
        epoch_loss /= len(train_loader)
        test_loss /= len(test_loader)

        epoch_accuracy /= len(train_loader)
        test_accuracy /= len(test_loader)
        
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/val", test_loss, epoch)
        writer.add_scalar("Acc/train", epoch_accuracy, epoch)
        writer.add_scalar("Acc/val", test_accuracy, epoch)
        writer.add_scalar("Recall/train", train_recall, epoch)
        writer.add_scalar("Recall/val", val_recall, epoch)
        writer.add_scalar("Precision/train", train_precision, epoch)
        writer.add_scalar("Precision/val", val_precision, epoch)

        train_losses.append(epoch_loss)
        val_losses.append(test_loss)
        train_accs.append(epoch_accuracy)
        val_accs.append(test_accuracy)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)

        model_name = f"weight_{weight}_epoch{epoch:02d}_acc{test_accuracy:.4f}_prec{val_precision:.4f}_rec{val_recall:.4f}.pth"
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))
        model_df.loc[len(model_df)] = {
        'Model': model_name,
        'weight': weight,
        'epoch': epoch,
        'acc': test_accuracy,
        'precision': val_precision,
        'recall': val_recall
    }

        epoch_end_time = time.time()
        epoch_used_time = epoch_end_time - epoch_start_time
        print(f'{weight}-epoch {epoch} done in {epoch_used_time} s')
        
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
    plt.savefig(os.path.join(img_dir,img_name) , dpi=300, bbox_inches="tight")
    plt.close()



    end_time = time.time()
    used_time = end_time - start_time
    print(f"Weight {weight} completed in {used_time} seconds")

    writer.close()

total_end_time = time.time()
total_used_time = total_end_time - total_start_time
total_used_time /= 60*60 
print(f"V1 completed in {total_used_time} hours")

model_df.to_csv(os.path.join(model_dir,'model_df.csv'), index=False)
