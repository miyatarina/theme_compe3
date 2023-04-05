# theme_compe3

# 概要
`train.py`でいくつかのモデルを学習した後，`ensemble.py`を用いて出力ファイルのアンサンブルを行う．
リーダーボードで最高点を記録した際に使用したモデルは以下の3つである．

### モデル1
| パラメータ | 実際の値 |
| ---- | --- |
| model | luke-japanese-large |
| seed | 5 |
| learning_rate | 3e-5 |
| batch_size | 16 |
| gradient_accumulation_steps | 4 |
| epochs | 100 |
| early_stopping_patience | 3 |