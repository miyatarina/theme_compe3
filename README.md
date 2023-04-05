# theme_compe3

# 概要
`train.py`でいくつかのモデルを学習した後，`ensemble.py`を用いて出力ファイルのアンサンブルを行う．
リーダーボードで最高点（0.647）を記録した際に使用したモデルは以下の3つである．

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

### モデル2
| パラメータ | 実際の値 |
| ---- | --- |
| model | luke-japanese-large |
| seed | 10 |
| learning_rate | 3e-5 |
| batch_size | 16 |
| gradient_accumulation_steps | 4 |
| epochs | 100 |
| early_stopping_patience | 3 |

### モデル3
| パラメータ | 実際の値 |
| ---- | --- |
| model | luke-japanese-large |
| seed | 5 |
| learning_rate | 3e-5 |
| batch_size | 16 |
| gradient_accumulation_steps | 8 |
| epochs | 100 |
| early_stopping_patience | 3 |

これらの3つのモデルの出力の平均をとるこった．

# 実際の使用方法
```
bash exp_003.sh
```
を実行することによってモデルの学習を行う．
パラメータを変更する場合は，`exp_003.sh`の中身を書き換えることで変更する．
パラメータを変更していくつかのモデルからそれぞれ出力ファイルを得た後，`ensemble.py`でアンサンブルを行う．

### ensemble.pyの説明
まず`def make_list`に出力ファイルのパスと空のリストを引数として与え，ファイルの中身をリストに追加する．
その後，アンサンブル結果を出力するファイルのパスを指定し，使用したいリストに対応してforループを書き換える．
