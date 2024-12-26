#!/bin/bash

# ローカルのフォルダが存在しない場合は作成
mkdir -p /app/logs /app/models

# 現在時刻をファイル名に使用
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 学習ログの記録
python train.py > /app/logs/train_${TIMESTAMP}.log 2>&1

# モデル保存
MODEL_PATH="/app/models/model_${TIMESTAMP}.pt"
python -c "import torch; from model import TransformerModel; model = TransformerModel(...); torch.save(model.state_dict(), '${MODEL_PATH}')"

echo "Training completed. Logs saved to /app/logs/train_${TIMESTAMP}.log"
echo "Model saved to ${MODEL_PATH}"
