from pathlib import Path
import string
import torch

from typing import List

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
)

import pytorch_lightning as pl

# Constants
MODEL_NAME = 't5-small'
LEARNING_RATE = 0.0001
SOURCE_MAX_TOKEN_LEN = 512
TARGET_MAX_TOKEN_LEN = 64
SEP_TOKEN = '<sep>'
TOKENIZER_LEN = 32101 #after adding the new <sep> token

# Model
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



class DistractorGenerator:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

        # Add SEP token and update tokenizer
        num_added_tokens = self.tokenizer.add_tokens([SEP_TOKEN])
        self.tokenizer_len = len(self.tokenizer)

        # Load checkpointed model
        checkpoint_path = Path("app/ml_models/distractor_generation/models/race-distractors.ckpt")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.dg_model = QGModel.load_from_checkpoint(checkpoint_path)
        self.dg_model.model.resize_token_embeddings(self.tokenizer_len)
        self.dg_model.eval()
        self.dg_model.freeze()

    def generate(self, generate_count: int, correct: str, question: str, context: str) -> list[str]:
        num_batches = (generate_count // 3) + 1
        output_text = self._model_predict(num_batches, correct, question, context)

        # Replace special tokens and extract distractors
        cleaned = output_text.replace('<pad>', '').replace('</s>', SEP_TOKEN)
        cleaned = self._replace_all_extra_ids(cleaned)

        distractors = cleaned.split(SEP_TOKEN)[:-1]
        return [
            d.translate(str.maketrans('', '', string.punctuation)).strip()
            for d in distractors if d.strip()
        ]

    @torch.no_grad()
    def _model_predict(self, generate_count: int, correct: str, question: str, context: str) -> str:
        prompt = f"{correct} {SEP_TOKEN} {question} {SEP_TOKEN} {context}"

        encoding = self.tokenizer(
            prompt,
            max_length=SOURCE_MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        generated_ids = self.dg_model.model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            num_beams=generate_count,
            num_return_sequences=generate_count,
            max_length=TARGET_MAX_TOKEN_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )

        return ''.join([
            self.tokenizer.decode(gid, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            for gid in generated_ids
        ])

    def _replace_all_extra_ids(self, text: str) -> str:
        while '<extra_id_' in text:
            start_idx = text.find('<extra_id_')
            end_idx = text.find('>', start_idx)
            if start_idx == -1 or end_idx == -1:
                break
            text = text[:start_idx] + SEP_TOKEN + text[end_idx + 1:]
        return text
