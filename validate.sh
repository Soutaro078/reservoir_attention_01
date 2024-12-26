#!/bin/bash

# 現在時刻をファイル名に使用
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 検証ログの記録
python validate.py > /app/logs/validate_${TIMESTAMP}.log 2>&1

echo "Validation completed. Logs saved to /app/logs/validate_${TIMESTAMP}.log"
