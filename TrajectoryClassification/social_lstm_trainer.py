import os
import pickle
from datetime import datetime
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class SocialLSTMDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        with open(pkl_path, "rb") as f:
            self.clips = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        positions = torch.tensor(clip["positions"], dtype=torch.float32)    # [T, N, 2]
        labels = torch.tensor(clip["labels"], dtype=torch.long)            # [N]
        
        # 應用資料增強 (如果有的話)
        if self.transform:
            positions = self.transform(positions)
            
        return positions, labels


class PositionTransform:
    """簡單的資料增強：隨機旋轉和平移"""
    def __init__(self, rotation_range=30, translation_range=5):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        
    def __call__(self, positions):
        # positions: [T, N, 2]
        # 1. 隨機旋轉
        angle = np.random.uniform(-self.rotation_range, self.rotation_range) * np.pi / 180
        rot_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=torch.float32)
        
        # 2. 隨機平移
        tx = np.random.uniform(-self.translation_range, self.translation_range)
        ty = np.random.uniform(-self.translation_range, self.translation_range)
        translation = torch.tensor([tx, ty], dtype=torch.float32)
        
        # 應用變換
        T, N, _ = positions.shape
        positions_flat = positions.reshape(-1, 2)  # [T*N, 2]
        positions_rot = torch.matmul(positions_flat, rot_matrix.t())  # [T*N, 2]
        positions_trans = positions_rot + translation  # [T*N, 2]
        
        return positions_trans.reshape(T, N, 2)


class AttentionSocialPooling(nn.Module):
    """增強版社交池化層：使用注意力機制權重化鄰居影響"""
    def __init__(self, radius=50, feat_dim=2, attention_dim=16):
        super().__init__()
        self.radius = radius
        self.attention = nn.Sequential(
            nn.Linear(feat_dim * 2, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, positions):
        # positions: [B, T, N, 2]
        B, T, N, C = positions.shape
        device = positions.device
        
        # 計算速度特徵 (可選)
        if T > 1:
            velocities = positions[:, 1:] - positions[:, :-1]
            velocities = torch.cat([velocities[:, 0:1], velocities], dim=1)  # 首幀速度複製
        else:
            velocities = torch.zeros_like(positions)
        
        # 計算位置差異
        pos_i = positions.unsqueeze(3)                   # [B, T, N, 1, 2]
        pos_j = positions.unsqueeze(2)                   # [B, T, 1, N, 2]
        diff = pos_j - pos_i                             # [B, T, N, N, 2]
        dist = torch.norm(diff, dim=-1)                  # [B, T, N, N]
        
        # 鄰居遮罩
        neighbor_mask = (dist > 0) & (dist < self.radius)  # [B, T, N, N]
        neighbor_mask = neighbor_mask.float().unsqueeze(-1)  # [B, T, N, N, 1]
        
        # 計算注意力權重
        # 將自身特徵和相對位置結合來計算注意力
        pos_i_expand = pos_i.expand(-1, -1, -1, N, -1)  # [B, T, N, N, 2]
        combined = torch.cat([pos_i_expand, diff], dim=-1)  # [B, T, N, N, 4]
        
        # 計算每對節點的注意力權重
        attention_input = combined.view(B * T * N * N, -1)  # [B*T*N*N, 4]
        attention_weights = self.attention(attention_input).view(B, T, N, N, 1)  # [B, T, N, N, 1]
        
        # 應用注意力權重和鄰居遮罩
        weighted_diff = diff * attention_weights * neighbor_mask  # [B, T, N, N, 2]
        
        # 聚合
        sum_weighted = weighted_diff.sum(dim=3)  # [B, T, N, 2]
        count = neighbor_mask.sum(dim=3).clamp(min=1e-6)  # [B, T, N, 1]
        social_feat = sum_weighted / count  # [B, T, N, 2]
        
        return social_feat


class SocialLSTMClassifier(nn.Module):
    def __init__(self, radius=50, input_dim=2, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.social_pooling = AttentionSocialPooling(radius=radius, feat_dim=input_dim)
        
        # 增加輸入特徵：位置(2) + 社交特徵(2) + 速度(2)
        self.lstm = nn.LSTM(input_dim * 3, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, positions):
        # positions: [B, T, N, 2]
        B, T, N, C = positions.shape
        device = positions.device

        # 計算速度
        if T > 1:
            velocities = positions[:, 1:] - positions[:, :-1]
            # 首幀速度設為0
            zeros = torch.zeros(B, 1, N, C, device=device)
            velocities = torch.cat([zeros, velocities], dim=1)  # [B, T, N, 2]
        else:
            velocities = torch.zeros_like(positions)  # [B, T, N, 2]

        # 社交池化
        social_feat = self.social_pooling(positions)  # [B, T, N, 2]
        
        # 組合特徵：位置 + 社交特徵 + 速度
        inp = torch.cat([positions, social_feat, velocities], dim=-1)  # [B, T, N, 6]
        inp = inp.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # [B*N, T, 6]
        
        # LSTM處理
        out, (hn, _) = self.lstm(inp)  # hn[-1]: [B*N, hidden_dim]
        
        # 分類層
        logits = self.fc(hn[-1])  # [B*N, 2]
        return logits.view(B, N, -1)  # [B, N, 2]


class EarlyStopping:
    """早停機制"""
    def __init__(self, patience=7, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        score = -val_loss  # 希望loss越低越好
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class Trainer:
    def __init__(self, model, optimizer, criterion, writer=None, scheduler=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.scheduler = scheduler
        self.best_model_path = 'best_model.pth'
        self.early_stopping = EarlyStopping(patience=200, path=self.best_model_path)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        
        pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
        for positions, labels in pbar:
            positions = positions.to(self.device)  # [B, T, N, 2]
            labels = labels.to(self.device)        # [N]
            B, T, N, _ = positions.shape

            self.optimizer.zero_grad()
            logits = self.model(positions)         # [B, N, 2]
            logits_flat = logits.view(-1, 2)       # [B*N, 2]
            labels_flat = labels.view(-1)          # [N] since B=1
            loss = self.criterion(logits_flat, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            preds = logits_flat.argmax(dim=1)
            correct += (preds == labels_flat).sum().item()
            total += labels_flat.size(0)
            total_loss += loss.item() * labels_flat.size(0)
            
            # 收集預測和標籤，用於計算指標
            all_preds.append(preds.cpu())
            all_labels.append(labels_flat.cpu())
            
            pbar.set_postfix(loss=loss.item(), acc=correct/total)

        avg_loss = total_loss / total
        avg_acc = correct / total
        
        # 計算更多指標
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        cm = confusion_matrix(all_labels, all_preds)
        
        if self.writer:
            self.writer.add_scalar("Loss/Train", avg_loss, epoch)
            self.writer.add_scalar("Acc/Train", avg_acc, epoch)
            # 添加混淆矩陣
            fig = self._plot_confusion_matrix(cm, ["safe", "dangerous"])
            self.writer.add_figure("CM/Train", fig, epoch)
        
        return avg_loss, avg_acc, cm

    def validate(self, loader, epoch):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        
        pbar = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)
        with torch.no_grad():
            for positions, labels in pbar:
                positions = positions.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(positions)
                logits_flat = logits.view(-1, 2)
                labels_flat = labels.view(-1)
                loss = self.criterion(logits_flat, labels_flat)

                preds = logits_flat.argmax(dim=1)
                correct += (preds == labels_flat).sum().item()
                total += labels_flat.size(0)
                total_loss += loss.item() * labels_flat.size(0)
                
                # 收集預測和標籤
                all_preds.append(preds.cpu())
                all_labels.append(labels_flat.cpu())
                
                pbar.set_postfix(loss=loss.item(), acc=correct/total)

        avg_loss = total_loss / total
        avg_acc = correct / total
        
        # 計算指標
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=["safe", "dangerous"])
        print(f"\nValidation Report (Epoch {epoch}):")
        print(report)
        
        if self.writer:
            self.writer.add_scalar("Loss/Val", avg_loss, epoch)
            self.writer.add_scalar("Acc/Val", avg_acc, epoch)
            fig = self._plot_confusion_matrix(cm, ["safe", "dangerous"])
            self.writer.add_figure("CM/Val", fig, epoch)
        
        # 早停檢查
        self.early_stopping(avg_loss, self.model)
        
        return avg_loss, avg_acc, cm
    
    def _plot_confusion_matrix(self, cm, class_names):
        """繪製混淆矩陣"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        
        # 添加刻度
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # 在每個格子中顯示數值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        return fig

    def fit(self, train_loader, val_loader, epochs=20):
        print(f"Training on device: {self.device}")
        best_val_acc = 0
        
        for ep in range(1, epochs+1):
            tr_loss, tr_acc, tr_cm = self.train_epoch(train_loader, ep)
            val_loss, val_acc, val_cm = self.validate(val_loader, ep)
            
            print(f"Epoch {ep}/{epochs} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Train CM:\n{tr_cm}\nVal CM:\n{val_cm}")
            
            if self.scheduler:
                self.scheduler.step(val_loss)
                
            # 檢查是否早停
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        # 載入最佳模型
        self.model.load_state_dict(torch.load(self.best_model_path))
        return self.model

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def run_training(
    pkl_path="social_lstm_clips.pkl",
    batch_size=4,  # 增加批次大小
    epochs=30,
    lr=1e-3,
    hidden_dim=128,
    num_layers=2,  # 增加層數
    dropout=0.2,
    use_data_augmentation=True
):
    # 1) 載入整個 dataset
    transform = PositionTransform() if use_data_augmentation else None
    dataset = SocialLSTMDataset(pkl_path, transform=transform)
    
    # --- 新增：計算 clip-level label & clip-level 權重
    clip_labels = []
    for _, labels in dataset:
        # labels: Tensor([0,0,1,0,...]) → 如果至少有一個 1，就視為危險
        clip_labels.append(int(labels.sum().item() > 0))
    counts_clip = Counter(clip_labels)   # e.g. {0: 200, 1: 20}
    # 權重反比於數量：危險 clip 較少，要打高一點
    clip_weights = [1.0 / counts_clip[l] for l in clip_labels]
    print(f"Clip class distribution: {counts_clip}")
    
    # 2) 隨機切 train / val
    n_train = int(len(dataset)*0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    # --- 新增：從 train_ds.indices 挑出對應的 clip_weights，給 sampler
    train_indices = train_ds.indices  # Subset 才有 .indices
    train_weights = [ clip_weights[i] for i in train_indices ]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    
    # 3) DataLoader：train 用 sampler，val 照常 shuffle=False
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              sampler=sampler,
                              num_workers=4)  # 使用多進程加速
    val_loader   = DataLoader(val_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4)

    # 4) 節點級別 class_weight → 仍舊放到同一 device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_labels = []
    for _, labels in dataset:
        all_labels += labels.tolist()
    counts = Counter(all_labels)   # 節點總數：{0: 10000, 1: 500}
    node_weight = torch.tensor([1.0/counts[i] for i in [0,1]],
                               dtype=torch.float, device=device)
    print(f"Node class distribution: {counts}")
    
    # 可以考慮使用 Focal Loss 來進一步處理不平衡問題
    criterion = nn.CrossEntropyLoss(weight=node_weight)
    criterion = criterion.to(device)
    
    # 5) 創建 model / optimizer / scheduler / writer
    model = SocialLSTMClassifier(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # 使用權重衰減(weight decay)幫助正則化
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 更靈活的學習率調整
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    # tensorboard
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 記錄超參數
    writer.add_text("Hyperparameters", 
                   f"batch_size={batch_size}, lr={lr}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, dropout={dropout}, "
                   f"data_augmentation={use_data_augmentation}")

    # 6) 傳 device 給 Trainer，並跑 fit
    trainer = Trainer(model, optimizer, criterion,
                     writer, scheduler, device=device)
    trainer.fit(train_loader, val_loader, epochs=epochs)
    writer.close()

    # save
    date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(f"weights/{date}", exist_ok=True)
    save_path = f"weights/{date}/social-lstm.pth"
    trainer.save_model(save_path)
    print(f"Model saved to {save_path}")
    return save_path


def run_inference(model_path, pkl_path="social_lstm_clips.pkl"):
    # load model
    model = SocialLSTMClassifier(num_layers=2)  # 注意: 確保與訓練時相同的參數
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # load data
    dataset = SocialLSTMDataset(pkl_path)
    loader = DataLoader(dataset, batch_size=1)

    results = []
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for idx, (positions, labels) in enumerate(tqdm(loader, desc="Inference")):
            logits = model(positions)
            preds = logits.argmax(dim=-1).squeeze(0)  # [N]
            results.append((idx, preds.tolist()))
            
            # 收集用於評估的資料
            all_preds.append(preds)
            all_labels.append(labels.squeeze(0))
    
    # 匯總結果
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # 計算指標
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["safe", "dangerous"])
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # 顯示每個clip中的危險節點
    print("\nDetailed Results:")
    for idx, pred in results:
        dangerous_nodes = [i for i, v in enumerate(pred) if v == 1]
        if dangerous_nodes:  # 只顯示有危險節點的clip
            print(f"Clip {idx}: Dangerous nodes = {dangerous_nodes}")
    
    return results, cm, report


def hyperparameter_search(pkl_path="social_lstm_clips.pkl"):
    """簡單的超參數搜索"""
    from sklearn.model_selection import ParameterGrid
    
    # 超參數網格
    param_grid = {
        'batch_size': [4, 8],
        'lr': [1e-3, 5e-4],
        'hidden_dim': [64, 128],
        'num_layers': [1, 2],
        'dropout': [0.1, 0.2]
    }
    
    grid = ParameterGrid(param_grid)
    results = []
    
    for params in grid:
        print(f"\nTrying parameters: {params}")
        try:
            model_path = run_training(
                pkl_path=pkl_path,
                batch_size=params['batch_size'],
                lr=params['lr'],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                epochs=10  # 減少epoch數來加速搜索
            )
            
            # 簡單評估
            _, cm, report = run_inference(model_path, pkl_path)
            
            # 從報告中提取F1 score
            import re
            f1_match = re.search(r'weighted avg\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)', report)
            f1_score = float(f1_match.group(1)) if f1_match else 0
            
            results.append((params, f1_score))
            print(f"F1 Score: {f1_score}")
            
        except Exception as e:
            print(f"Error with params {params}: {e}")
    
    # 找出最佳參數
    results.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 3 parameter sets:")
    for i, (params, score) in enumerate(results[:3]):
        print(f"{i+1}. Score: {score:.4f}, Params: {params}")
    
    return results[0][0] if results else None


if __name__ == '__main__':
    # 超參數搜索（可選）
    # best_params = hyperparameter_search("Annotation_App/social_lstm_clips_0.pkl")
    # if best_params:
    #    print(f"Best parameters found: {best_params}")
    #    model_path = run_training("Annotation_App/social_lstm_clips_0.pkl", **best_params, epochs=100)
    
    # 直接訓練（使用改進的默認參數）
    model_path = run_training("Annotation_App/social_lstm_clips_0.pkl", 
                              batch_size=1, 
                              epochs=100, 
                              num_layers=2,
                              dropout=0.2,
                              use_data_augmentation=True)
    
    # 執行推理和評估
    #run_inference(model_path, "Annotation_App/social_lstm_clips_0.pkl")