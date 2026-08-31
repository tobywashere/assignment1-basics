from collections.abc import Iterable, Iterator
from pickle import load
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
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    input_string = 'the cat<|endoftext|>ate'

    tokenizer = Tokenizer(vocab, merges, ['<|endoftext|>'])

    encoded_text = tokenizer.encode(input_string)

    print(f"encoded_text = {encoded_text}")

    decoded_text = tokenizer.decode(encoded_text)

    print(f"decoded_text = {decoded_text}")

    assert(decoded_text == input_string)
