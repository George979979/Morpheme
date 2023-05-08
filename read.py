
import random

def to_BMES(morphemes):
    answer = []
    for morpheme in morphemes:
        if len(morpheme) > 1 and ":" in morpheme:
            morpheme, morph_type = morpheme.split(":")
        else:
            morph_type = None
        if len(morpheme) == 1:
            curr_answer = ["S"]
        else:
            curr_answer = ["B"] + ["M"] * (len(morpheme) - 2) + ["E"]
        if morph_type is not None:
            curr_answer = [label + "-" + morph_type for label in answer]
        answer += curr_answer
    return answer


def to_BIO(morphemes, morph_type=None):
    answer = []
    for morpheme in morphemes:
        if len(morpheme) > 1 and ":" in morpheme:
            morpheme, morph_type = morpheme.split(":")
        else:
            morph_type = None
        curr_answer = ["B"] + ["I"] * (len(morpheme) - 1)
        if morph_type is not None:
            curr_answer = [label + "-" + morph_type for label in answer]
        answer += curr_answer
    return answer


def read_infile(infile, sep="\t", morph_sep="/", variant_sep=",", language=None):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            word, analysis = line.split(sep)
            analysis = analysis.strip().split(variant_sep)[0]
            morphemes = [x for x in analysis.split(morph_sep) if x != ""]
            line_data = {"word": word, "morphemes": morphemes, "bmes_labels": to_BMES(morphemes),
                         "bio_labels": to_BIO(morphemes)}
            answer.append(line_data)
    return answer


def read_wordlist(infile, n=None, min_length=5, language=None):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            if " " in line:
                line = line.split()[-1]
            line = line.strip()
            if len(line) >= min_length and all(x.isalpha() or x == "-" for x in line):
                answer.append(line)
    random.shuffle(answer)
    if n is not None:
        answer = answer[:n]
    answer = [{"word": word} for word in answer]
    return answer

