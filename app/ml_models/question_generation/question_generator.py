from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
import pytorch_lightning as pl

# Constants
MODEL_NAME = 't5-small'
LEARNING_RATE = 0.0001
SOURCE_MAX_TOKEN_LEN = 300
TARGET_MAX_TOKEN_LEN = 80
SEP_TOKEN = '<sep>'
TOKENIZER_LEN = 32101 #after adding the new <sep> token

# QG Model
class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.model.resize_token_embeddings(TOKENIZER_LEN) #resizing after adding new tokens to the tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)


class QuestionGenerator:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        _ = self.tokenizer.add_tokens([SEP_TOKEN])
        self.tokenizer_len = len(self.tokenizer)

        checkpoint_path = Path('app/ml_models/question_generation/models/multitask-qg-ag.ckpt')
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.qg_model = QGModel.load_from_checkpoint(checkpoint_path)
        self.qg_model.model.resize_token_embeddings(self.tokenizer_len)
        self.qg_model.eval()
        self.qg_model.freeze()

    def generate(self, answer: str, context: str) -> str:
        output = self._model_predict(answer, context)
        return output.split(SEP_TOKEN)[-1].strip()

    def generate_qna(self, context: str) -> Tuple[str, str]:
        masked_answer = '[MASK]'
        output = self._model_predict(masked_answer, context)
        parts = output.split(SEP_TOKEN)

        if len(parts) < 2:
            return '', parts[0].strip()
        return parts[0].strip(), parts[1].strip()

    @torch.no_grad()
    def _model_predict(self, answer: str, context: str) -> str:
        prompt = f"{answer} {SEP_TOKEN} {context}"
        encoding = self.tokenizer(
            prompt,
            max_length=SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        generated_ids = self.qg_model.model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            num_beams=16,
            max_length=TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        decoded_outputs = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]

        return ' '.join(decoded_outputs)
