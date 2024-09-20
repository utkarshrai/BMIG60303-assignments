import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_excel('./All_Articles_Excel_Dec2019July2020.xlsx')
df = df.sample(frac=0.1, random_state=42)
df.replace('NaN', pd.NA).dropna(axis=1)
df = df[['Date Added', 'Author', 'Title', 'Abstract', 'Year']]
df = df.dropna()
df['Abstract'] = df['Abstract'].str.lower()


def part01(df):
    vectorizer = TfidfVectorizer()
    df.dropna(inplace=True)
    tfidf_matrix = vectorizer.fit_transform(df['Abstract'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=vectorizer.get_feature_names_out())

    return tfidf_df


tfidf_df = part01(df)
print("TF -IDF dataframe created. The datafrane is giant. Printing first 10 columns and 10 rows")
print(tfidf_df[tfidf_df.columns[:10]].head())
print()


def part02(tfidf_df):
    top_words_co = []
    for i in range(10):
        vector = tfidf_df.iloc[i]
        top_words = vector.sort_values(ascending=False).head(5)
        top_words_l = [(word, score) for word, score in top_words.items()]
        top_words_co.append(top_words_l)

    return top_words_co


top_words = part02(tfidf_df)
print("Top five words for each of the first ten abstracts")
print()
print(top_words)
print()

print("Part 3")


def part03(tfidf_df, query):
    vectorizer = TfidfVectorizer(vocabulary=tfidf_df.columns)
    query = vectorizer.fit_transform([query.lower()])
    cosine_similarities = cosine_similarity(query, tfidf_df).flatten()
    # tfidf_df.sort_values('similarity', ascending=False)[:5]
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]
    for i in top_5_indices:
        print("Cosine Similarity of " +
              str(cosine_similarities[i]) + "at index "+str(i))
    return top_5_indices


query = "virus spread health"
top_Index = part03(tfidf_df, query)
# print(list(df.iloc[op_docs.tolist()]['Abstract']))
