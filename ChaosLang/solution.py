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
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class LanguageDataset(Dataset):
    def __init__(self, data : pd.DataFrame, tokenizer : Tokenizer, max_seq_len):
        self.max_seq_len = max_seq_len
        self.pad_idx = tokenizer.get_vocab().get("[PAD]")
        
        self.data = data
        self.encodings = []
        self.language_dict = {}

        for i, row in data.iterrows():
            self.encodings.append(tokenizer.encode(str(row["pronunciation"])).ids)

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
    
class PhoneticsDataset: 
    def __init__(self, data : pd.DataFrame, tokenizer : Tokenizer, max_seq_len, batch_size : int):
        self.datasets : dict[str, Dataset] = {}
        self.dataloaders : dict[str, DataLoader]  = {}

        for language in data["language"].value_counts().keys():
            print("Loading", language)
            self.datasets[language] = LanguageDataset(data[data["language"] == language], tokenizer, max_seq_len)
            self.dataloaders[language] = DataLoader(self.datasets[language], batch_size, shuffle=True)

    def get_data_loaders(self):
        return self.dataloaders

from transformers import GPT2LMHeadModel, GPT2Model, GPT2Config
class PhoneticTransformer(nn.Module):
    def __init__(self, vocab, seq_len, languages = {}):
        super().__init__()
        self.eos_token_id = vocab["[EOS]"]
        self.max_len = seq_len

        self.gpts = nn.ModuleList([GPT2LMHeadModel(GPT2Config(
            vocab_size=len(vocab),
            n_positions=seq_len,
            n_embd=72,
            n_layer = 4, 
            n_head = 4, 
            bos_token_id=vocab["[BOS]"],
            eos_token_id=vocab["[EOS]"], 
            
        )) for _ in range(len(languages))])
        self.device = "cpu"
        self.languages = list(languages.keys())
        self.language_weights = []
        self.set_language_weights(languages)


    def set_language_weights(self, languages):
        self.language_weights = [torch.tensor(x, dtype=torch.float32) for x in languages.values()]
        total_weight = sum(self.language_weights)
        self.language_weights = [weight / total_weight for weight in self.language_weights]

    def get_language_gpt(self, language):
        return self.gpts[self.languages.index(language)]

    def forward(self, x):
        output_logits = []
        for i, gpt in enumerate(self.gpts): 
            gpt_out = gpt.forward(x)
            logit = self.language_weights[i] * gpt_out.logits

            output_logits.append(logit)

        output = torch.stack(output_logits, dim = 0)
        output = torch.sum(output, dim=0)
        return output
    
    def generate(self,  
        input_ids = [],
        pad_token_id = None,
        no_repeat_ngram_size=0,
        do_sample = True,
        top_k=50,
        top_p=0.95,
        temperature=1.0
    ):
        curr_seq : torch.Tensor = input_ids 

        while True: 
            outputs= []
            for i, gpt in enumerate(self.gpts):
                gpt : GPT2LMHeadModel = gpt
                output = gpt.generate(
                    curr_seq, 
                    pad_token_id = pad_token_id,
                    max_length=curr_seq.shape[1] + 1,  # Maximum length of the generated text
                    no_repeat_ngram_size=no_repeat_ngram_size,  # Prevent repetition
                    do_sample = do_sample,
                    top_k=top_k,  # Limits the sampling pool to top_k tokens
                    top_p=top_p,  # Cumulative probability for nucleus sampling
                    temperature=temperature,  # Adjust the randomness of predictions,
                    output_logits = True,
                    return_dict_in_generate= True
                )

                outputs.append(self.language_weights[i] * output.logits[0][0])

            output = torch.stack(outputs, dim = 0)
            next_token_scores = torch.sum(output, dim=0)
            
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

            curr_seq = torch.concat([curr_seq, next_token], axis= 1)
            # finished sentences should have their next token be a padding token
            if self.eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                if next_token == self.eos_token_id:
                    break 

            if len(curr_seq) >= self.max_len:
                break


        return curr_seq

    def to_device(self, device):
        self.to(device)
        self.device = device
        return self