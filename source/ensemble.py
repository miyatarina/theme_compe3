import collections

seed1 = list()
seed5 = list()
seed10 = list()
seed35 = list()
seed42 = list()
bert = list()
accum8_seed5 = list()


def make_list(file_name, list_name):
    with open(file_name, "r") as f:
        for line in f:
            list_name.append(line.strip())

make_list("/home/miyata/python/workspace/theme_competition3/submission/luke_large_seed1.csv", seed1)
make_list("/home/miyata/python/workspace/theme_competition3/submission/luke_large_seed5.csv", seed5)
make_list("/home/miyata/python/workspace/theme_competition3/submission/luke_large_seed10.csv", seed10)
make_list("/home/miyata/python/workspace/theme_competition3/submission/luke_large_seed35.csv", seed35)
make_list("/home/miyata/python/workspace/theme_competition3/submission/luke_large_seed42.csv", seed42)
make_list("/home/miyata/python/workspace/theme_competition3/submission/tohoku_BERT_weight_kappa.csv", bert)
make_list("/home/miyata/python/workspace/theme_competition3/submission/luke_large_accum8_seed5.csv", accum8_seed5)

# LUKE_largeのみ 
# 多数決 : 0.614
with open("/home/miyata/python/workspace/theme_competition3/submission/luke_large_max-vote.csv", "w") as f:
    for a, b, c, d, e in zip(seed1, seed5, seed10, seed35, seed42):
        votes = [a, b, c, d, e]
        max_vote = collections.Counter(votes).most_common()[0][0]
        f.write(max_vote + "\n")

# 平均 LB : 0.627
with open("/home/miyata/python/workspace/theme_competition3/submission/luke_large_avg.csv", "w") as f:
    for a, b, c, d, e in zip(seed1, seed5, seed10, seed35, seed42):
        avg = (float(a) + float(b) + float(c) + float(d) + float(e)) / 5
        f.write(str(round(avg)) + "\n")


# LUKE_large + BERT
# 多数決 : 0.620
with open("/home/miyata/python/workspace/theme_competition3/submission/luke_large_bert_max-vote.csv", "w") as f:
    for a, b, c, d, e, g in zip(seed1, seed5, seed10, seed35, seed42, bert):
        votes = [a, b, c, d, e, g]
        max_vote = collections.Counter(votes).most_common()[0][0]
        f.write(max_vote + "\n")

# 平均 LB : 0.614
with open("/home/miyata/python/workspace/theme_competition3/submission/luke_large_bert_avg.csv", "w") as f:
    for a, b, c, d, e, g in zip(seed1, seed5, seed10, seed35, seed42, bert):
        avg = (float(a) + float(b) + float(c) + float(d) + float(e) + float(g)) / 6
        f.write(str(round(avg)) + "\n")


# LUKE_large_accum4_seed5 + LUKE_large_accum8_seed5
# 多数決 : 0.626
with open("/home/miyata/python/workspace/theme_competition3/submission/luke_large_accum8_and_seed5_max-vote.csv", "w") as f:
    for a, b in zip(seed5, accum8_seed5):
        votes = [a, b]
        max_vote = collections.Counter(votes).most_common()[0][0]
        f.write(max_vote + "\n")

# 平均 LB : 0.633
with open("/home/miyata/python/workspace/theme_competition3/submission/luke_large_accum8_and_seed5_avg.csv", "w") as f:
    for a, b in zip(seed5, accum8_seed5):
        avg = (float(a) + float(b)) / 2
        f.write(str(round(avg)) + "\n")


# LUKE_large_accum4_seed5 + LUKE_large_accum8_seed5 + BERT
# 多数決 :0.617
with open("/home/miyata/python/workspace/theme_competition3/submission/luke_large_accum8_and_seed5_max-vote.csv", "w") as f:
    for a, b, c in zip(seed5, accum8_seed5, bert):
        votes = [a, b, c]
        max_vote = collections.Counter(votes).most_common()[0][0]
        f.write(max_vote + "\n")

# 平均 LB : 
with open("/home/miyata/python/workspace/theme_competition3/submission/luke_large_accum8_and_seed5_avg.csv", "w") as f:
    for a, b, c in zip(seed5, accum8_seed5, bert):
        avg = (float(a) + float(b) + float(c)) / 3
        f.write(str(round(avg)) + "\n")


# LUKE_large_accum4_seed5 + LUKE_large_accum8_seed5 + LUKE_large_accum4_seed10
# 平均 LB : 0.647
with open("/home/miyata/python/workspace/theme_competition3/submission/luke_large_accum8_and_seed5_seed10_avg.csv", "w") as f:
    for a, b, c in zip(seed5, accum8_seed5, seed10):
        avg = (float(a) + float(b) + float(c)) / 3
        f.write(str(round(avg)) + "\n")