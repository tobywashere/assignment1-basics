import argparse
from collections import Counter

text = """                                                                                                                                                                                                       
low low low low low                                                                                                                                                                                              
lower lower widest widest widest                                                                                                                                                                                 
newest newest newest newest newest newest                                                                                                                                                                        
"""

def merge_pair(tokens: tuple[int, ...],
               pair: tuple[int, int],
               new_token: int) -> tuple[int, ...]:
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

    return tuple(result)

def pretokenization(text, num_merges):
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    vocab[256] = '<|endoftext|>'.encode('utf')
    
    word_counts = Counter()
    for word in text.split():
        word_counts[tuple(word.encode('utf'))] += 1
        
    vocab_size = 257
    for new_token in range(vocab_size, vocab_size+num_merges):
        # 1. Count pairs of CURRENT token IDs
        pair_counts = Counter()

        for tokens, count in word_counts.items():
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] += count

        # 2. Pick best pair
        max_count = max(pair_counts.values())
        best_pair = None
        for pair in pair_counts:
            
        best_pair = max([(count, (vocab[pair[0]], vocab[pair[1]])) for pair, count in pair_counts.items()])[1]
        print(f"best_pair={best_pair}")

        # 3. Add its byte representation to vocab
        vocab[new_token] = best_pair[0] + best_pair[1]

        # 4. Replace that pair everywhere with new token ID
        word_counts = {
            merge_pair(tokens, vocab[new_token], new_token): count
            for tokens, count in word_counts.items()
        }

    return vocab, word_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_merges', type=int, default=6)
    args = parser.parse_args()

    vocab, word_counts = pretokenization(text, args.num_merges)
    print(f"vocab={vocab}")
    print(f"word_counts={word_counts}")
