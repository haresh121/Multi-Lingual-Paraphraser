import pandas as pd
import os
import json


files_train = {
    "msr_train": "../data/MSR/modified/msr_train.csv",
    "ppdb": "../data/PPDB-2.0/modified/ppdb-2.0-mod.csv",
    "tapaco": "../data/TaPaCo/modified/tapaco-mod.csv"
}
files_valid = {
    "msr_test": "../data/MSR/modified/msr_test.csv"
}
clean_trained = []
clean_valid = []
for file in files_train:
    if file == "tapaco":
        break
    print(files_train[file])
    lines = list(open(files_train[file]).readlines())[1:]
    try:
        for line in lines:
            temp = line.split('|||')
            clean_trained.append({
                "phrase": temp[0],
                "paraphrase": temp[1].strip()
            })
    except IndexError:
        pass

for file in files_valid:
    if file == "tapaco":
        break
    print(files_valid[file])
    lines = list(open(files_valid[file]).readlines())[1:]
    try:
        for line in lines:
            temp = line.split('|||')
            clean_valid.append({
                "phrase": temp[0],
                "paraphrase": temp[1].strip()
            })
    except IndexError:
        pass

with open(files_train["tapaco"]) as f:
    lines = list(f.readlines())[1:]
    iid, n_line = lines[0].split("|||")
    for n, line in enumerate(lines):
        sp = line.split("|||")
        if iid == int(sp[0]):
            clean_trained.append({
                "phrase": n_line,
                "paraphrase":sp[1]
            })
        elif iid != int(sp[0]):
            n_line = sp[1]

train = pd.DataFrame(clean_trained)
valid = pd.DataFrame(clean_valid)

train.to_csv("../data/clean-data/train.csv", '|', index=False)
valid.to_csv("../data/clean-data/valid.csv", '|', index=False)

with open('../data/clean-data/train.json', "w+") as f:
    json.dump(clean_trained, f)

with open('../data/clean-data/valid.json', "w+") as f:
    json.dump(clean_trained, f)
