import re
import collections
import pandas as pd


def get_vocab(filename):
    vocab = collections.Counter()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            words = [element.lower() for element in words]
            for word in words:
                vocab[" ".join(list(word))] += 1  #Divide each word into characters
    return vocab


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq  #Count the frequency of each character pair
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")  #Match target character pairs
    for word in v_in:
        w_out = p.sub("".join(pair), word)  #Replace the target character pair with a new character
        v_out[w_out] = v_in[word]

    return v_out


def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens


def main(count, num_merges, vocab):

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if len(pairs) == 0:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

    new_vocab = {}
    for item in vocab.items():
        new_key = item[0].replace(' ', ' ##')
        new_vocab[new_key] = item[1]

    tokens = get_tokens(new_vocab)
    tokens = sorted(tokens.items(), key=lambda d: d[1], reverse=True)
    vocab_size_list.append(len(tokens))   # num_merges = %s  vocab_size = %s

    num = num_merges*(count+1)
    if len(pairs) != 0:
        with open('/home/wangping/make_vocab/result/vocab%s.txt' % num, 'w') as file:
            for item in tokens:
                file.write(item[0] + '\n')
    return vocab_size_list, vocab


if __name__ == '__main__':
    vocab = get_vocab("/home/wangping/make_vocab/raw_input_file.txt")  
    vocab_size_list = []
    for count in range(255):
        num_merges = 50
        vocab_size_list, vocab = main(count, num_merges, vocab)

    # #Export Excel
    df = pd.DataFrame(list(vocab_size_list))
    df.to_excel('result/vocab_size_list.xlsx')




