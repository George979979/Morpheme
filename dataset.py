from collections import Counter

import numpy as np
from torch.utils.data.dataset import Dataset


class BertSegmentDatasetReader(Dataset):

    def __init__(self, data, d, vocab=None, embeddings=None,
                 letters=None, labels=None,
                 min_symbol_count=1, max_morpheme_length=None, field="bio_labels"):
        self.data = data
        self.vocab = vocab
        self.embeddings = embeddings
        self.d = d
        if max_morpheme_length is None or max_morpheme_length <= d:
            self.max_morpheme_length = d
        else:
            self.max_morpheme_length = max_morpheme_length
        self.field = field
        # symbols
        if letters is None:
            letters = Counter(x for elem in data for x in elem["word"])
            self.letters_ = ["PAD", "UNK"] + [x for x, count in letters.items() if count >= min_symbol_count]
        else:
            self.letters_ = letters
        self.letter_codes_ = {label: i for i, label in enumerate(self.letters_)}
        if labels is None:
            labels = ["PAD"] + list({x for elem in data for x in elem[self.field]})
        self.labels_ = labels
        self.label_codes_ = {label: i for i, label in enumerate(self.labels_)}

    def _word_to_matrix(self, word):
        d = self.d + int(self.max_morpheme_length > self.d)
        subtoken_indexes = np.zeros(shape=(len(word), 2 * d), dtype=int)
        for start in range(len(word)):
            long_morpheme_index, long_morpheme_length = None, None
            for length in range(1, self.max_morpheme_length + 1):
                if start + length > len(word):
                    break
                ngram = word[start:start + length]
                if start > 0:
                    ngram = "##" + ngram
                index = self.vocab.get(ngram)
                if index is not None:
                    if length <= d:
                        subtoken_indexes[start, length - 1] = index
                        if start + length < len(word):
                            subtoken_indexes[start + length, d + length - 1] = index
                    else:
                        long_morpheme_index, long_morpheme_length = index, length
            if long_morpheme_index is not None:
                subtoken_indexes[start, d - 1] = long_morpheme_index
                if start + long_morpheme_length < len(word) and subtoken_indexes[
                    start + long_morpheme_length, 2 * d - 1] == 0:
                    subtoken_indexes[start + long_morpheme_length, 2 * d - 1] = long_morpheme_index
        answer = self.embeddings[subtoken_indexes]
        answer[subtoken_indexes == 0] = 0.0
        return subtoken_indexes, answer

    def __getitem__(self, i):
        word = self.data[i]["word"]
        answer = {"letters": [self.letter_codes_.get(letter, 1) for letter in word]}
        if self.vocab is not None and self.embeddings is not None:
            subtoken_indexes, x = self._word_to_matrix(word)
            answer["inputs"] = x
            answer["subtoken_indexes"] = subtoken_indexes
        try:
            answer["y"] = [self.label_codes_[label] for label in self.data[i][self.field]]
        except:
            pass
        mask = self.data[i].get("mask")
        mask = np.array(mask, dtype="float") if mask is not None else np.ones(shape=(len(word),), dtype="float")
        answer["mask"] = mask * self.data[i].get("weight", 1.0)
        return answer

    def __len__(self):
        return len(self.data)

    @property
    def output_dim(self):
        return 2 * (self.d + int(self.max_morpheme_length > self.d))


class LetterDatasetReader(Dataset):

    def __init__(self, data, letters=None, min_symbol_count=1, field="bio_labels"):
        self.data = data
        self.field = field
        # symbols
        if letters is None:
            letters = Counter(x for elem in data for x in elem["word"])
            self.letters_ = ["PAD", "UNK"] + [x for x, count in letters.items() if count >= min_symbol_count]
        else:
            self.letters_ = letters
        self.letter_codes_ = {label: i for i, label in enumerate(self.letters_)}

    def __getitem__(self, i):
        word = self.data[i]["word"]
        letters = [self.letter_codes_.get(letter, 1) for letter in word]
        next_letters = letters[1:] + [0]
        prev_letters = [0] + letters[:-1]
        return {
            "letters": letters, "next_letter": next_letters, "prev_letter": prev_letters
        }

    def __len__(self):
        return len(self.data)
