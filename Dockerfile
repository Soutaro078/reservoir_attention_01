# ベースイメージ（GPUサポート付きPyTorch）
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 必要パッケージをインストール
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev wget git \
    && apt-get clean

RUN ln -s /usr/bin/python3 /usr/bin/python

# Pythonライブラリのインストール
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# コードをコピー
COPY . /app

# 作業ディレクトリ
WORKDIR /app

# ローカルフォルダをコンテナにマウント
VOLUME ["/app/logs", "/app/models", "/app/data"]

CMD ["bash"]
