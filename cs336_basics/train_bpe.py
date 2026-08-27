import argparse
from collections import Counter, defaultdict
from functools import reduce
from itertools import repeat, starmap
from multiprocessing import Pool
import pathlib
import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)

class PairCounter:
    def __init__(self):
        self.pair_counts = Counter()
        self.token_map = defaultdict(set)

def bpe_tokenizer(input_path: str,
                  vocab_size: int,
                  special_tokens: list[str],
                  num_processes=6):
    assert(len(special_tokens) > 0)
    joined_special_tokens = '|'.join([re.escape(st) for st in special_tokens])

    with open(input_path, "rb") as f:
        # Use the first special token to find chunk boundaries
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode('utf-8'))

    if num_processes > 1:
        with Pool(num_processes) as p:
            word_count_map = p.starmap(word_count, zip(boundaries[:-1], boundaries[1:], repeat(input_path), repeat(joined_special_tokens)))
    else:
        word_count_map = starmap(word_count, zip(boundaries[:-1], boundaries[1:], repeat(input_path), repeat(joined_special_tokens)))

    word_counts = reduce(merge_word_counts, word_count_map)

    vocab = {}
    BYTE_MAX = 256
    assert(vocab_size > BYTE_MAX + len(special_tokens))
    for i in range(BYTE_MAX):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[BYTE_MAX + i] = special_tokens[i].encode('utf-8')

    merges = []

    pretokenization(word_counts, vocab, vocab_size, merges)
    
    return vocab, merges

def word_count(start, end, input_path, joined_special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    word_counts = Counter()

    for text in re.split(joined_special_tokens, chunk):
        for word in COMPILED_PAT.finditer(text):
            word_counts[tuple(word.group().encode('utf-8'))] += 1

    return word_counts

def merge_word_counts(word_count1, word_count2):
    for word, count in word_count2.items():
        word_count1[word] += count
    return word_count1

def pretokenization(word_counts, vocab, vocab_size, merges):
    # 1. Count pairs of CURRENT token IDs
    pair_counts = PairCounter()
    for tokens, count in word_counts.items():
        for pair in zip(tokens, tokens[1:]):
            pair_counts.pair_counts[pair] += count
            pair_counts.token_map[pair].add(tokens)

    for new_token in range(len(vocab), vocab_size):        
        # 2. Pick best pair
        best_pair = pick_best_pair(pair_counts.pair_counts, vocab)
        if best_pair is None:
            break
        
        # 3. Add its byte representation to vocab
        vocab[new_token] = (
            vocab[best_pair[0]]
            + vocab[best_pair[1]]
        )
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 4. Replace that pair everywhere with new token ID
        for tokens in list(pair_counts.token_map[best_pair]):
            count = word_counts[tokens]
            del word_counts[tokens]
            new_tokens = merge_pair(tokens, best_pair, new_token, pair_counts, count)
            word_counts[new_tokens] = count
        assert(not best_pair in pair_counts.pair_counts)
        assert(not best_pair in pair_counts.token_map)

def pick_best_pair(pair_counts, vocab):
    best_pair = None
    if len(pair_counts) == 0:
        return best_pair
    max_count = max(pair_counts.values())
    for key in pair_counts:
        if pair_counts[key] < max_count:
            continue
        if best_pair is None:
            best_pair = key
            continue
        old_pair = (vocab[best_pair[0]], vocab[best_pair[1]])
        new_pair = (vocab[key[0]], vocab[key[1]])
        best_pair = key if new_pair > old_pair else best_pair
    return best_pair

def merge_pair(tokens: tuple[int, ...],
               pair: tuple[int, int],
               new_token: int,
               pair_counts: PairCounter,
               count: int) -> tuple[int, ...]:
    result = []
    i = 0

    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]
        ):
            result.append(new_token)
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    for i in range(len(tokens)-1):
        old_pair = (tokens[i], tokens[i+1])
        pair_counts.pair_counts[old_pair] -= count
        assert(pair_counts.pair_counts[old_pair] >= 0)
        pair_counts.token_map[old_pair].discard(tokens)
        if pair_counts.pair_counts[old_pair] == 0:
            del pair_counts.pair_counts[old_pair]
            del pair_counts.token_map[old_pair]

    new_tokens = tuple(result)
    for i in range(len(result)-1):
        new_pair = (result[i], result[i+1])
        pair_counts.pair_counts[new_pair] += count
        pair_counts.token_map[new_pair].add(new_tokens)

    return new_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=pathlib.Path)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--special_tokens', type=list[str], default=['<|endoftext|>'])
    parser.add_argument('--num_processes', type=int, default=6)
    args = parser.parse_args()

    vocab, merges = bpe_tokenizer(args.input_path, args.vocab_size, args.special_tokens, args.num_processes)
    print(f"vocab={vocab}")
    print(f"merges={merges}")
    
    
    
