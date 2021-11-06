from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tqdm.auto import tqdm


normalizer = normalizers.Sequence([NFD(), StripAccents()])

print(normalizer.normalize_str("Héllò hôw are ü?"))

data = {
    "msr_train": "/Users/haresh/Haresh/Paraphrasers/data/MSR/msr_paraphrase_train.tsv",
    "msr_test": "/Users/haresh/Haresh/Paraphrasers/data/MSR/msr_paraphrase_test.tsv",
}

for i in data:
    file = f'/Users/haresh/Haresh/Paraphrasers/data/MSR/modified/msr_{i.split("_")[-1]}.csv'
    print(file)
    f_mod = open(file, 'w+')
    with open(data[i]) as f:
        temp = f.readlines()
        for line in tqdm(temp):
            sp = line.split("\t")
            a = normalizer.normalize_str(sp[3])
            b = normalizer.normalize_str(sp[4])
            f_mod.write(f"{a}|||{b}\n")
    f_mod.close()

with open("../data/PPDB-2.0/ppdb-2.0-m-phrasal") as f:
    f_mod = open("../data/PPDB-2.0/modified/ppdb-2.0-mod.csv", "w+")
    f_mod.write("S1|||S2\n")
    lines = list(f.readlines())
    for line in tqdm(lines):
        s = line.split("|||")
        f_mod.write(f"{s[1]}|||{s[2]}\n")
    f_mod.close()

with open("../data/TaPaCo/tapaco_v1.0/en.txt") as f:
    lines = list(f.readlines())
    f_mod = open("../data/TaPaCo/modified/tapaco-mod.csv", "w+")
    f_mod.write("ID|||Sentence\n")
    for line in tqdm(lines):
        s = line.split("\t")
        f_mod.write(f"{s[0]}|||{s[2]}\n")
    f_mod.close()