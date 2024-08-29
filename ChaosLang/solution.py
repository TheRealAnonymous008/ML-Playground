from tokenizers import Tokenizer, models, processors, decoders, pre_tokenizers

import pandas as pd 
import numpy as np
from languages import LANGUAGES

""" CharacterTokenzier for Hugging Face Transformers.

This is heavily inspired from CanineTokenizer in transformers package.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

class CharToks():
    def __init__(self, ids):
        self.ids = ids

class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, **kwargs):
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[UNK]": 5,
            "[EOS]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def get_vocab(self):
        return self._vocab_str_to_int

    def _tokenize(self, text: str) -> List[str]:
        return list(text)
    
    def encode(self, text : str):
        toks = [self.bos_token] + self._tokenize(text) + [self.eos_token]
        return CharToks(self.convert_tokens_to_ids(toks))

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 100):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PhoneticTransformer(nn.Module):
    def __init__(self, vocab_size, d_model = 64, n_head = 8, n_decoders = 4, d_feedforward = 512, max_seq_len = 100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward= d_feedforward),
            num_layers= n_decoders,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, tgt_mask = None):
        tgt_embedded = self.embedding(tgt).permute(1, 0, 2) 
        tgt_embedded = self.positional_encoding.forward(tgt_embedded) 
        transformer_output = self.transformer.forward(
            src=tgt_embedded,
            mask = tgt_mask,
            is_causal=True
        )
        output = self.fc_out(transformer_output)
        return output
    
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PhoneticsDataset(Dataset):
    def __init__(self, encodings, vocab : dict, max_seq_Len = 100):
        self.encodings = encodings
        self.max_seq_len = max_seq_Len
        self.pad_idx = vocab.get("[PAD]")

    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        tgt_idx = len(encoding) - 1
        input_ids = encoding[: tgt_idx]
        input_ids = input_ids + [self.pad_idx] * (self.max_seq_len - len(input_ids))

        target_ids = encoding[1 : tgt_idx + 1]
        target_ids = target_ids + [self.pad_idx] * (self.max_seq_len - len(target_ids))

        input_ids = torch.tensor(input_ids, dtype = torch.long)
        target = torch.tensor(target_ids, dtype=torch.long)

        return input_ids, target