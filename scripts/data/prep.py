import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import T5Tokenizer
import hydra
from omegaconf import DictConfig
from cleantext import clean as CLEAN
from tqdm.auto import tqdm

PATH = os.getcwd()


class CustomDataset(Dataset):
    def __init__(self, **kwargs):
        super(CustomDataset, self).__init__()
        self.name = kwargs.get('name')
        self.files = kwargs.get('path')
        self.sep, self.s1, self.s2 = kwargs.get('d_args')
        print(self.sep, self.s1, self.s2)
        self.tokenizer = T5Tokenizer.from_pretrained(kwargs.get('tokenizer_name_or_path'))
        self.max_length = kwargs.get('max_length')

        self.source = []
        self.target = []

        self.build_data()


    def __len__(self):
        return len(self.source)


    def __getitem__(self, idx):
        self.source_ids = self.source[idx]["input_ids"].squeeze()
        self.target_ids = self.target[idx]["input_ids"].squeeze()

        self.input_mask = self.source[idx]["attention_mask"].squeeze()
        self.target_mask = self.target[idx]["attention_mask"].squeeze()

        return {
            "source_ids": self.source_ids,
            "target_ids": self.target_ids,
            "source_mask": self.input_mask,
            "target_mask": self.target_mask
        }


    def build_data(self):
        args = dict(
            fix_unicode=True,
            to_ascii=True,
            lower=False,
            no_urls=True,
            no_currency_symbols=True,
            replace_with_url="<URL>",
            replace_with_currency_symbol="<CUR>"
        )
        for n, file in enumerate(self.files):
            print(f"Processing File : {PATH+'/'+file}, num : {n}")
            with open(PATH+"/"+file) as f:
                data = list(f.readlines())[1:]
            for line in tqdm(data):
                temp = line.split(self.sep[n])
                source, target = CLEAN(temp[self.s1[n]], **args), CLEAN(temp[self.s2[n]], **args)

                source = "paraphrase: " + source
                target = target

                tokenized_source = self.tokenizer.batch_encode_plus(
                    [source], max_length=self.max_length, padding="max_length", return_tensors="pt", truncation=True
                )
                tokenized_target = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.max_length, padding="max_length", return_tensors="pt", truncation=True
                )

                self.source.append(tokenized_source)
                self.target.append(tokenized_target)


@hydra.main(config_path="../../conf", config_name="config")
def prepare_data(cfg: DictConfig):
    ret = {"trainData": "", "validationData": "", "testData": ""}
    for split in ["train-files", "validation-files", "test-files"]:
        test = cfg['data-args'].get(split, None)
        if test:
            name = split.split("-")[0]+"Data"
            print(name)
            args = dict(
                name=name,
                path=[],
                d_args=[[], [], []],
                tokenizer_name_or_path=cfg['model-args']['tokenizer_name_or_path'],
                max_length=cfg['model-args']['max_seq_length']
            )
            for path in cfg['data-args'][split]:
                file_args = cfg['data-args'][split][path]
                args['path'].append(path)
                args['d_args'][0].append(file_args[0])
                args['d_args'][1].append(file_args[1])
                args['d_args'][2].append(file_args[2])
            ret[name] = CustomDataset(**args)

    return ret


if __name__ == "__main__":
    ret = prepare_data()
    print(ret)