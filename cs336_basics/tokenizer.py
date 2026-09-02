import argparse
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
import pathlib
from pickle import load
import regex as re
import numpy as np

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None, num_processes=6):
        self.vocab = vocab
        self.reverse_vocab = {value: key for key, value in vocab.items()}
        self.merges = {merge: order for order, merge in enumerate(merges)} 
        self.special_tokens = special_tokens
        if special_tokens:
            sorted_st = sorted(special_tokens, key=len, reverse=True)
            self.joined_special_tokens = re.compile(f"({'|'.join([re.escape(st) for st in sorted_st])})")
            new_token = len(vocab)
            for st in special_tokens:
                encoded_st = st.encode('utf-8')
                if not encoded_st in self.reverse_vocab:
                    self.vocab[new_token] = encoded_st
                    self.reverse_vocab[encoded_st] = new_token
                    new_token += 1
        self.num_processes = num_processes
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.COMPILED_PAT = re.compile(PAT)
        self.already_seen = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, num_processes=6):
        with open(vocab_filepath, 'rb') as f:
            vocab = load(f)
        with open(merges_filepath, 'rb') as f:
            merges = load(f)
        return cls(vocab, merges, special_tokens, num_processes)

    def encode(self, txt: str) -> list[int]:
        if self.special_tokens:
            chunks = self.joined_special_tokens.split(txt)
        else:
            chunks = [txt]
        result = []
        for chunk in chunks:
            match = self.joined_special_tokens.match(chunk) if self.special_tokens else None
            if match:
                result.append(self.reverse_vocab[match.group().encode('utf-8')])
                continue
            for word in self.COMPILED_PAT.finditer(chunk):
                result += self._bpe_encode(word.group().encode('utf-8'))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        if self.num_processes > 1:
            with Pool(self.num_processes) as p:
                for ids in p.imap(self.encode, iterable, chunksize=10**5):
                    yield from ids
        else:
            for text in iterable:
                yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        result = bytes()
        for id in ids:
            result += self.vocab[id]
        return result.decode('utf-8', errors="replace")

    def _bpe_encode(self, word: bytes) -> list[bytes]:
        if word in self.already_seen:
            return list(self.already_seen[word])
        merged_word = [word[i:i+1] for i in range(len(word))]

        while True:
            min_pair = None
            min_pri = float('inf')
            for i in range(len(merged_word)-1):
                pair = (merged_word[i], merged_word[i+1]) 
                if pair in self.merges and self.merges[pair] < min_pri:
                    min_pri = self.merges[pair]
                    min_pair = pair
            if min_pair is None:
                break            
            merged_word = self._apply_merge(merged_word, min_pair)
            
        result = [self.reverse_vocab[token] for token in merged_word]
        self.already_seen[word] = result
        return result

    def _apply_merge(self, word, merge):
        i = 0
        result = []
        while i < len(word):
            if i+1 < len(word) and word[i] == merge[0] and word[i+1] == merge[1]:
                result.append(merge[0] + merge[1])
                i += 2
            else:
                result.append(word[i])
                i += 1
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_path', type=pathlib.Path)
    parser.add_argument('merge_path', type=pathlib.Path)
    parser.add_argument('--special_tokens', type=list[str], default=['<|endoftext|>'])
    parser.add_argument('--encode_path', type=pathlib.Path, default=None)
    parser.add_argument('--num_processes', type=int, default=6)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merge_path, args.special_tokens, args.num_processes)

    if args.encode_path:
        with open(args.encode_path, 'r') as f:
            ids = tokenizer.encode_iterable(f)
            np_ids = np.fromiter(ids, dtype=np.uint16)
            np.save(args.encode_path.stem + "_ids", np_ids)
        
