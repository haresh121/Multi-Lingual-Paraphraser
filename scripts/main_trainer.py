from transformers import Trainer, TrainingArguments
from transformers import T5Model, T5ForConditionalGeneration, AutoTokenizer
import wandb
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers import decoders
import pandas as pd
import torch
import json

wandb.init(project="tester", entity="codeblack")

model = T5ForConditionalGeneration.from_pretrained("t5-small")

args = TrainingArguments(
    report_to="wandb",
    output_dir="../output",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.001,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=1e-4
)

class ParaSentences(Dataset):
    def __init__(self, tok_file = "../tokenizers/paraBert.json", pdata="../data/clean-data/train.json"):
        super(ParaSentences, self).__init__()
        self.tokenizer = Tokenizer.from_file(tok_file)
        self.tokenizer.decoder = decoders.WordPiece()
        self.tokenizer.enable_padding(pad_token="<pad>", pad_id=self.tokenizer.token_to_id("<pad>"), length=50)
        self.decoder_input_ids = self.tokenizer.encode("<pad>").ids

        with open(pdata) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        phrase = self.tokenizer.encode(self.data[idx]['phrase']).ids
        pphrase = self.tokenizer.encode(self.data[idx]['paraphrase']).ids

        return {
            "input_ids": torch.tensor(phrase),
            "decoder_input_ids": self.decoder_input_ids,
            "labels": torch.tensor(pphrase)
        }

train_ds = ParaSentences()
val_ds = ParaSentences(pdata="../data/clean-data/valid.json")


trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)

trainer.train()