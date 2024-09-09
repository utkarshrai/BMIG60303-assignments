import pandas as pd
import re
import nltk
from nltk.util import ngrams
from collections import Counter
nltk.download('punkt')

df=pd.read_excel('/workspaces/BMIG60303-assignments/All_Articles_Excel_Dec2019July2020.xlsx')
df = df[['Date Added', 'Author', 'Title', 'Abstract', 'Year']]
pattern = r'\bItaly|Italian\b'
print("Part 1")
print("Absctract truncated to first 50 chars")

matches_l = []
for index, row in df.iterrows():
    abstract = row['Abstract']
    if isinstance(abstract, str):
        matches = re.findall(pattern, abstract, re.IGNORECASE)
        match_count = len(matches)
        if match_count > 0:
            matches_l.append((matches, match_count, abstract))
top_matches = sorted(matches_l, key=lambda x: x[1], reverse=True)[:10]

for i in top_matches:
    print(i[0])
    print(i[1])
    print(i[2][:50]+'....')

print()
print("Part 2")

from nltk.tokenize import wordpunct_tokenize

bigrams = Counter()
pattern = re.compile(r'\b(Italy|Italian)\b', re.IGNORECASE)
for index, row in df.iterrows():
    abstract = row['Abstract']
    if isinstance(abstract, str):
        tokens = wordpunct_tokenize(abstract)
        bigrams_list = list(ngrams(tokens, 2))
        re_search = [bigram for bigram in bigrams_list if pattern.search(bigram[0])]
        bigrams.update(re_search)

common_bigrams = bigrams.most_common(20)
for i in common_bigrams:
    print(str(i[0])+"\t"+ str(i[1]))

print()
print("Part 3")
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
bigrams = Counter()
pattern = re.compile(r'\b(Italy|Italian)\b', re.IGNORECASE)
for index, row in df.iterrows():
    abstract = row['Abstract']
    if isinstance(abstract, str):
        tokens = wordpunct_tokenize(abstract)
        cleaned_tokens = [token for token in tokens if token.isalnum() and token.lower() not in stop_words]
        bigrams_list = list(ngrams(cleaned_tokens, 2))
        re_search = [bigram for bigram in bigrams_list if pattern.search(bigram[0])]
        bigrams.update(re_search)
common_bigrams = bigrams.most_common(20)
for i in common_bigrams:
    print(str(i[0])+"\t"+ str(i[1]))