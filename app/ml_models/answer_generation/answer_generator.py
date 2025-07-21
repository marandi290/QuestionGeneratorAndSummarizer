from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
)

# Constants
MODEL_NAME = 't5-small'
SOURCE_MAX_TOKEN_LEN = 64
TARGET_MAX_TOKEN_LEN = 24
LEARNING_RATE = 1e-4


class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor = None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        loss, _ = self(**batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self(**batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self(**batch)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)


class AnswerGenerator:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

        checkpoint_path = Path("app/ml_models/answer_generation/models/squad-answer-generation.ckpt")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        self.ag_model = QGModel.load_from_checkpoint(checkpoint_path)
        self.ag_model.eval()
        self.ag_model.freeze()

    def generate(self, context: str, generate_count: int) -> list[str]:
        return self._model_predict(context, generate_count)

    @torch.no_grad()
    def _model_predict(self, context: str, generate_count: int) -> list[str]:
        encoding = self.tokenizer(
            context,
            max_length=SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        generated_ids = self.ag_model.model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            num_beams=generate_count,
            num_return_sequences=generate_count,
            max_length=TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )

        outputs = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]

        return outputs
