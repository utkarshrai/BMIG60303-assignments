from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from langdetect import detect
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import string
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
print("Part 1")
data = pd.read_excel(
    '/workspaces/BMIG60303-assignments/All_Articles_Excel_Dec2019July2020.xlsx')

df = data[['Abstract']].dropna()
df['Abstract'] = df['Abstract'].astype(str)


def is_english(text):
    try:
        return detect(text) == 'en'
    except BaseException:
        return False


df['English'] = df['Abstract'].apply(is_english)
random_abstracts_df = df[df['English']].sample(n=5000, random_state=0)
stops = stopwords.words('english') + list(string.punctuation)
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stops)
X = vectorizer.fit_transform(random_abstracts_df['Abstract'])
lda = LatentDirichletAllocation(n_components=10, random_state=0)
lda.fit(X)


def get_topic_words(lda_model, feature_names, n_top_words=10):
    topics = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        topic_words = [feature_names[i]
                       for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[f"Topic {topic_idx + 1}"] = topic_words
    return topics


feature_names = vectorizer.get_feature_names_out()
topics = get_topic_words(lda, feature_names)
for topic, words in topics.items():
    print(f'{topic}: {", ".join(words)}')
vocab = vectorizer.get_feature_names_out()
term_frequency = X.sum(axis=0).A1
doc_lengths = X.sum(axis=1).A1
lda_vis = pyLDAvis.prepare(
    topic_term_dists=lda.components_,
    doc_topic_dists=lda.transform(X),
    doc_lengths=doc_lengths,
    vocab=vocab,
    term_frequency=term_frequency
)

print()
print("Part 2")


def fit_and_print_lda(X, n_topics, vectorizer, n_top_words=10):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        print(
            f"Topic {
                topic_idx +
                1}:" +
            " ".join(
                [
                    feature_names[i] for i in topic.argsort()[
                        :-
                        n_top_words -
                        1:-
                        1]]))


tokenizer = TreebankWordTokenizer()
custom_stopwords = set(stops).union(
    {'covid', 'coronavirus', 'sars', 'cov', '19', 'doi'})
custom_stopwords_list = list(custom_stopwords)

tfidf_vectorizer_custom_stopwords = TfidfVectorizer(
    stop_words=custom_stopwords_list,
    tokenizer=tokenizer.tokenize
)

X_tfidf_custom_stopwords = tfidf_vectorizer_custom_stopwords.fit_transform(
    random_abstracts_df['Abstract'])

print("LDA with 10 Topics (Custom Stopwords & Tokenizer):")
fit_and_print_lda(X_tfidf_custom_stopwords, n_topics=10,
                  vectorizer=tfidf_vectorizer_custom_stopwords)
print()

tfidf_vectorizer_fewer_features = TfidfVectorizer(
    stop_words=custom_stopwords_list,
    tokenizer=tokenizer.tokenize,
    max_features=2000
)

X_tfidf_fewer_features = tfidf_vectorizer_fewer_features.fit_transform(
    random_abstracts_df['Abstract'])


print("\nLDA with 10 Topics (Fewer Features): ")
fit_and_print_lda(X_tfidf_fewer_features, n_topics=10,
                  vectorizer=tfidf_vectorizer_fewer_features)

tfidf_vectorizer_no_stopwords = TfidfVectorizer(
    tokenizer=tokenizer.tokenize,
    max_features=5000
)

X_tfidf_no_stopwords = tfidf_vectorizer_no_stopwords.fit_transform(
    random_abstracts_df['Abstract'])

print("\nLDA with 10 Topics (No Stopwords): ")
fit_and_print_lda(
    X_tfidf_no_stopwords,
    n_topics=10,
    vectorizer=tfidf_vectorizer_no_stopwords)

print("\nLDA with 10 Topics (Default Stopwords and Tokenizer): ")
vectorizer_default = TfidfVectorizer(
    max_df=0.95,
    min_df=5,
    stop_words='english',
    max_features=5000)
X_default = vectorizer_default.fit_transform(random_abstracts_df['Abstract'])
fit_and_print_lda(X_default, n_topics=10, vectorizer=vectorizer_default)

print()
print("Part 3: ")
pyLDAvis.save_html(lda_vis, 'lda_part3.html')
print("Visualisation Saved As lda_part3.html")
