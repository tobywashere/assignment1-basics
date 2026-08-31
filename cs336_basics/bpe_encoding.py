import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)

def bpe_encode(word, vocab, merges):
    merged_word = [word[i:i+1] for i in range(len(word))]
    for merge in merges:
        merged_word = apply_merge(merged_word, merge)

    return [vocab[token] for token in merged_word]

def apply_merge(word, merge):
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
    input_string = 'the cat ate'

    reverse_vocab = {value: key for key, value in vocab.items()}

    result = []

    for word in COMPILED_PAT.finditer(input_string):
        result += bpe_encode(word.group().encode('utf-8'), reverse_vocab, merges)

    print(result)
    
