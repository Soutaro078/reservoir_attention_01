import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from dataset import TransformerDataset
from model import TransformerModel
import os

	# •	fpath: データセット（.npy ファイル）のパス。
	# •	enc_seq_len: エンコーダに入力するシーケンス長。
	# •	target_seq_len: デコーダで予測するシーケンス長。
	# •	d_obs: 観測データの次元数（入力データの特徴量数）。
	# •	d_model: Transformer内部の隠れ層の次元数。
	# •	num_heads: マルチヘッドアテンションのヘッド数。
	# •	enc_num_layers: エンコーダの層数。
	# •	dec_num_layers: デコーダの層数。
	# •	num_epochs: トレーニングのエポック数。
	# •	batchsize: バッチサイズ。
	# •	enc_dropout: エンコーダのドロップアウト率。
	# •	dec_dropout: デコーダのドロップアウト率。
	# •	learning_rate: エポックごとに設定できる学習率の辞書。

def train(
    fpath: str,
    enc_seq_len: int,
    target_seq_len: int,
    d_obs: int,
    d_model: int,
    num_heads: int,
    enc_num_layers: int,
    dec_num_layers: int,
    num_epochs: int,
    batchsize: int,
    enc_dropout: int = .2,
    dec_dropout: int = .2,
    learning_rate: dict = {0: 1e-3},
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = TransformerDataset(fpath, enc_seq_len, target_seq_len)
    train_loader = DataLoader(train_dataset, batchsize)#ここでバッチ単位でデータを取得することができる
    enc_mask = train_dataset.enc_mask
    dec_mask = train_dataset.dec_mask

    model = TransformerModel(
        seq_len=enc_seq_len+target_seq_len,
        d_obs=d_obs,
        d_model=d_model,
        num_heads=num_heads,
        enc_num_layers=enc_num_layers,
        dec_num_layers=dec_num_layers,
        enc_dropout=enc_dropout,
        dec_dropout=dec_dropout,
    )

    optimizer = torch.optim.Adam(params=model.parameters())
    loss_fn = nn.MSELoss()
    train_length = len(dataset)

    history = {"train_loss": []}
    n = 0
    train_loss = 0


    for epoch in range(num_epochs):
        # バッチごとの処理進捗を視覚的に表示。
        epoch_iterator = tqdm(train_loader)
        sum_loss = 0
        # エポックごとに学習率を変更する場合の処理。
        if epoch in learning_rate.keys():
            optimizer.lr = learning_rate[epoch]
            print("Changed learning rate to:", optimizer.lr)
        for (enc_input, dec_input, target) in epoch_iterator:
            # epoch += 1
            loss = 0
            optimizer.zero_grad()

            output = model(enc_input, dec_input, enc_mask, dec_mask)
            loss = loss_fn(output, target)
            sum_loss += loss.item()
            epoch_iterator.set_description(f"Loss={loss.item()}")

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {sum_loss/len(train_loader)}")
        

    return model


if __name__ == "__main__":
    print('start')
    model = train(
        '/app/data/lorenz63_test.npy',
        enc_seq_len=10,
        target_seq_len=4,
        d_obs=3,
        d_model=512,
        num_heads=8,
        enc_num_layers=4,
        dec_num_layers=4,
        num_epochs=5,
        batchsize=512,
        enc_dropout=0.4,
        dec_dropout=0.4
    )

    # models ディレクトリが存在しない場合は作成
    os.makedirs('./models', exist_ok=True)
    torch.save(model, f"./models/model_{datetime.now()}.pt")
