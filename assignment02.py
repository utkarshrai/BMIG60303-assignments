import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import spacy
from nltk.corpus import stopwords
from collections import Counter
import string
nltk.download('wordnet', quiet=True)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def part01():
    consume = wn.synset('consume.v.02')
    hyponyms = consume.closure(lambda s: s.hyponyms())
    return set(hyponyms)


def part02(synset_set):
    lemmas = {lemma for synset in synset_set for lemma in synset.lemma_names()}
    filtered_lemmas = {lemma for lemma in lemmas if '_' not in lemma}
    return filtered_lemmas


def part03(texts, consume_terms):
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    num_bigrams = Counter()
    for text in texts:
        doc = nlp(text)
        tokens = [
            token.lemma_.lower() for token in doc
            if token.lemma_.lower() not in stop_words and token.lemma_ not in punctuation
        ]
        bigrams = zip(tokens, tokens[1:])
        for first_word, second_word in bigrams:
            if first_word in consume_terms:
                num_bigrams[(first_word, second_word)] += 1
    return num_bigrams


result = part01()
print(result)
print(len(result))
print()


synsets = part01()
lemmas = part02(synsets)
print(lemmas)
print(len(lemmas))

print()
res = part03(pd.read_csv('1557tweets.csv').text, part02(part01()))
# res = part03(df['text'], consume_terms)

print(res.most_common(10))
