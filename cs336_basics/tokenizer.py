import argparse
from collections.abc import Iterable, Iterator
import pathlib
from pickle import load, dump
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.reverse_vocab = {value: key for key, value in vocab.items()}
        self.merges = merges
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
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.COMPILED_PAT = re.compile(PAT)
        self.already_seen = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = load(f)
        with open(merges_filepath, 'rb') as f:
            merges = load(f)
        return cls(vocab, merges, special_tokens)

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
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        result = bytes()
        for id in ids:
            result += self.vocab[id]
        return result.decode('utf-8', errors="replace")

    def _bpe_encode(self, word: bytes) -> list[bytes]:
        if word in self.already_seen:
            return self.already_seen[word]
        merged_word = [word[i:i+1] for i in range(len(word))]
        # TODO: O(words x merges) is inefficient, optimize
        for merge in self.merges:
            merged_word = self._apply_merge(merged_word, merge)
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
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merge_path, args.special_tokens)

    if args.encode_path:
        with open(args.encode_path, 'r') as f:
            ids = list(tokenizer.encode_iterable(f))
        with open(args.encode_path.stem + "_ids" + args.encode_path.suffix, 'wb') as f:
            dump(ids, f)
        
