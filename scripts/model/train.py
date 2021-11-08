import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from scripts.data.prep import generate_datasets
from torch.utils.data import DataLoader

import nltk
# nltk.download("punkt")
from nltk.tokenize import sent_tokenize

from transformers import AdamW, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

import os
import hydra
from omegaconf import DictConfig
import logging

PATH = os.getcwd()
CONF = None

@hydra.main(config_path=os.path.abspath("./conf"), config_name="config")
def get_config(cfg: DictConfig):
    global CONF
    CONF = cfg

class T5FineTuner(pl.LightningModule):
    def __init__(self, **kwargs):
        super(T5FineTuner, self).__init__()
        self.args = kwargs.copy()
        self.setup_logging()
        self.prepare_data()
        logging.info(f"T5FineTuner Initializing")
        self.model = T5ForConditionalGeneration.from_pretrained(kwargs.get('model_name_or_path', 't5-base'))

    def setup_logging(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=os.path.abspath(self.args['logging_file_path']), level=logging.INFO)

    def prepare_data(self) -> None:
        self.train_dataset = self.args['dataset']['train_dataset']
        self.valid_dataset = self.args['dataset']['valid_dataset']

    def forward(
            self,
            input_ids,
            attention_mask = None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels
        )

    def single_step(self, batch):
        lm_labels = batch['target_ids']
        lm_labels[lm_labels[:, :] == 0] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        logging.info(f"training started for batch: {batch_idx}")
        loss = self.single_step(batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        logging.info(f"validation started for batch: {batch_idx}")
        loss = self.single_step(batch)
        return {'val_loss': loss}

    def training_epoch_end(self, outputs) -> None:
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logging.info(f"training epoch end. average loss: {avg_train_loss}")
        return {"avg_train_loss": avg_train_loss}

    def validation_epoch_end(self, outputs) -> None:
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logging.info(f"validation epoch end. average loss: {avg_train_loss}")
        return {"avg_val_loss": avg_train_loss}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args['lr'], eps=self.args['adam_epsilon'])
        self.opt = optimizer
        return [optimizer]

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.args['train_batch_size'], drop_last=True, shuffle=True,
                                num_workers=4)

        t_total = (
                (len(dataloader.dataset) // (self.args['train_batch_size'] * max(1, self.args['n_gpu'])))
                // self.args['gradient_accumulation_steps']
                * float(self.args['nepochs'])
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.args['warmup_steps'], num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.valid_dataset, batch_size=self.args['valid_batch_size'], num_workers=4)
        return dataloader


if __name__ == "__main__":
    get_config()
    datasets = generate_datasets(CONF)
    model_args = CONF['model-args']
    seed_everything(model_args['seed'])
    args = dict(
        model_name_or_path=model_args['model_name_or_path'],
        logging_file_path=model_args['logging_file_path'],
        lr=model_args["learning_rate"],
        weight_decay=model_args['weight_decay'],
        adam_epsilon=model_args['adam_epsilon'],
        warmup_steps=model_args['warmup_steps'],
        train_batch_size=model_args['train_batch_size'],
        valid_batch_size=model_args['eval_batch_size'],
        num_train_epochs=model_args['num_train_epochs'],
        n_gpu=model_args['n_gpu'],
        max_grad_norm=model_args['max_grad_norm'],
        dataset={
            'train_dataset': datasets['trainData'],
            'valid_dataset': datasets['validationData']
        }
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.abspath(model_args['output_dir']), monitor="val_loss", mode="min", save_top_k=5
    )
    train_args = dict(
        accumulate_grad_batches=model_args.gradient_accumulation_steps,
        gpus=model_args.n_gpu,
        max_epochs=model_args.n_epochs,
        precision=16 if model_args.fp_16 else 32,
        gradient_clip_val=model_args.max_grad_norm,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback]
    )
    model = T5FineTuner(**args)
    trainer = pl.Trainer(**train_args)

    trainer.fit(model)