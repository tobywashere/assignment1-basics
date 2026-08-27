import argparse
from collections import Counter, defaultdict

text = """                                                                                                                                                                                                       
low low low low low                                                                                                                                                                                              
lower lower widest widest widest                                                                                                                                                                                 
newest newest newest newest newest newest
booook
"""

class PairCounter:
    def __init__(self):
        self.pair_counts = Counter()
        self.token_map = defaultdict(set)

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
        pair_counts.token_map[old_pair].remove(tokens)
        if pair_counts.pair_counts[old_pair] == 0:
            del pair_counts.pair_counts[old_pair]
            del pair_counts.token_map[old_pair]

    for i in range(len(result)-1):
        new_pair = (result[i], result[i+1])
        pair_counts.pair_counts[new_pair] += count
        pair_counts.token_map[new_pair].add(tuple(result))
        
    return tuple(result)

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

def pretokenization(text, num_merges):
    vocab = {}
    vocab_size  = 256
    for i in range(vocab_size):
        vocab[i] = bytes([i])
    vocab[vocab_size] = '<|endoftext|>'.encode('utf')
    vocab_size += 1
    
    word_counts = Counter()
    for word in text.split():
        word_counts[tuple(word.encode('utf'))] += 1

    # 1. Count pairs of CURRENT token IDs
    pair_counts = PairCounter()
    for tokens, count in word_counts.items():
        for pair in zip(tokens, tokens[1:]):
            pair_counts.pair_counts[pair] += count
            pair_counts.token_map[pair].add(tokens)

    for new_token in range(vocab_size, vocab_size+num_merges):        
        # 2. Pick best pair
        best_pair = pick_best_pair(pair_counts.pair_counts, vocab)
        if best_pair is None:
            break
        print(f"best_pair={best_pair}, count={pair_counts.pair_counts[best_pair]}, tokens to merge={pair_counts.token_map[best_pair]}")

        # 3. Add its byte representation to vocab
        vocab[new_token] = (
            vocab[best_pair[0]]
            + vocab[best_pair[1]]
        )

        # 4. Replace that pair everywhere with new token ID
        for tokens in list(pair_counts.token_map[best_pair]):
            count = word_counts[tokens]
            del word_counts[tokens]
            new_tokens = merge_pair(tokens, best_pair, new_token, pair_counts, count)
            word_counts[new_tokens] = count

        assert(not best_pair in pair_counts.pair_counts)
        assert(not best_pair in pair_counts.token_map)

    return vocab, word_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_merges', type=int, default=6)
    args = parser.parse_args()

    vocab, word_counts = pretokenization(text, args.num_merges)
    print(f"vocab={vocab}")
    print(f"word_counts={word_counts}")
