from dataclasses import dataclass

import pickle

@dataclass
class Language: 
    language_vocabulary : dict[str, str]

    def get_word_list(self):
        return [x for x in self.language_vocabulary.values()]


def save_language(language : Language, path  : str):
    with open(path, "wb") as fp: 
        pickle.dump(language.language_vocabulary, fp)



from tokenizers import Tokenizer
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace 
from tokenizers.processors import BertProcessing
from tokenizers.implementations import ByteLevelBPETokenizer
import os
def train_custom_tokenizer(language: Language, target_size : int, custom_tokenizer_path : str):
    words = language.get_word_list()

    os.makedirs("temp_data", exist_ok=True)
    with open("temp_data/text_corpus.txt", "w", encoding="utf-8") as f: 
        for line in words: 
            f.write(f"{line} ")

    custom_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    custom_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size = target_size, 
        min_frequency = 0,
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    custom_tokenizer.train(["temp_data/text_corpus.txt"], trainer = trainer)
    custom_tokenizer.save(custom_tokenizer_path)

    return custom_tokenizer