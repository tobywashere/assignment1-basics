import argparse
from collections import Counter
import pathlib
import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def bpe_tokenizer(input_path: str,
                  vocab_size: int,
                  special_tokens: list[str],
                  num_processes=6):
    joined_special_tokens = '|'.join([re.escape(st) for st in special_tokens])
    
    word_counts = Counter()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            for text in re.split(joined_special_tokens, chunk):
                for word in re.finditer(PAT, text):
                    word_counts[tuple(word.group().encode('utf-8'))] += 1

    vocab = {}
    BYTE_MAX = 256
    assert(vocab_size > BYTE_MAX)
    for i in range(BYTE_MAX):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[BYTE_MAX + i] = special_tokens[i].encode('utf-8')

    merges = []

    pretokenization(word_counts, vocab, vocab_size, merges)
    
    return vocab, merges

def pretokenization(word_counts, vocab, vocab_size, merges):
    # 1. Count pairs of CURRENT token IDs
    pair_counts = Counter()
    for tokens, count in word_counts.items():
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += count

    for new_token in range(len(vocab), vocab_size):        
        # 2. Pick best pair
        best_pair = pick_best_pair(pair_counts, vocab)
        if best_pair is None:
            break

        # 3. Add its byte representation to vocab
        vocab[new_token] = (
            vocab[best_pair[0]]
            + vocab[best_pair[1]]
        )
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 4. Replace that pair everywhere with new token ID
        word_counts = {
            merge_pair(tokens, best_pair, new_token, pair_counts, count): count
            for tokens, count in word_counts.items()
        }
        del pair_counts[best_pair]

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
               pair_counts: dict[tuple[int, int], int],
               count: int) -> tuple[int, ...]:
    result = []
    i = 0

    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]
        ):
            if i > 0:
                old_pair = (tokens[i-1], tokens[i])
                pair_counts[old_pair] -= count
                assert(pair_counts[old_pair] >= 0)
                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]
                new_pair = (result[-1], new_token)
                pair_counts[new_pair] += count
            result.append(new_token)
            i += 2
        else:
            if result and result[-1] == new_token:
                old_pair = (tokens[i-1], tokens[i])
                pair_counts[old_pair] -= count
                assert(pair_counts[old_pair] >= 0)
                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]
                new_pair = (result[-1], tokens[i])
                pair_counts[new_pair] += count
            result.append(tokens[i])
            i += 1

    return tuple(result)

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
    
    
    
