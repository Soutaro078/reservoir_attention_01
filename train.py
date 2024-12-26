import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime

from dataset import TransformerDataset  # データセットの読み込み
from model import TransformerModel  # Transformerモデルの読み込み

# GPU/CPUのデバイス設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 学習設定
ENC_SEQ_LEN = 10
TARGET_SEQ_LEN = 4
D_OBS = 3
D_MODEL = 512
NUM_HEADS = 8
ENC_NUM_LAYERS = 4
DEC_NUM_LAYERS = 4
NUM_EPOCHS = 5
BATCH_SIZE = 512
ENC_DROPOUT = 0.4
DEC_DROPOUT = 0.4
LEARNING_RATE = 1e-3

# データセットとデータローダー
train_dataset = TransformerDataset("/app/data/lorenz63_train.npy", ENC_SEQ_LEN, TARGET_SEQ_LEN)
val_dataset = TransformerDataset("/app/data/lorenz63_val.npy", ENC_SEQ_LEN, TARGET_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# モデルの構築
model = TransformerModel(
    seq_len=ENC_SEQ_LEN + TARGET_SEQ_LEN,
    d_obs=D_OBS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    enc_num_layers=ENC_NUM_LAYERS,
    dec_num_layers=DEC_NUM_LAYERS,
    enc_dropout=ENC_DROPOUT,
    dec_dropout=DEC_DROPOUT,
).to(device)

# 損失関数とオプティマイザ
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ログ保存用
history = {"train_loss": [], "val_loss": []}

# トレーニングループ
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for enc_input, dec_input, target in epoch_iterator:
        optimizer.zero_grad()

        # データをGPUに転送
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        target = target.to(device)

        # モデルの順伝播
        output = model(enc_input, dec_input)
        loss = loss_fn(output, target)

        # 誤差逆伝播と重み更新
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        epoch_iterator.set_description(f"Train Loss: {loss.item():.6f}")

    avg_train_loss = train_loss / len(train_loader)
    history["train_loss"].append(avg_train_loss)

    # 検証ループ
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for enc_input, dec_input, target in val_loader:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            target = target.to(device)

            output = model(enc_input, dec_input)
            loss = loss_fn(output, target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    history["val_loss"].append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

# モデルの保存
os.makedirs("/app/models", exist_ok=True)
model_save_path = f"/app/models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ログの保存
os.makedirs("/app/logs", exist_ok=True)
log_save_path = f"/app/logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
with open(log_save_path, "w") as log_file:
    for epoch, (train_loss, val_loss) in enumerate(zip(history["train_loss"], history["val_loss"])):
        log_file.write(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}\n")
print(f"Logs saved to {log_save_path}")
