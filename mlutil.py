import string
string.punctuation
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

#Stop words are words like “and”, “the”, “him”, which are presumed to be uninformative & non-predicative , so avoided
stopwords.words("english")[100:110]                   # to see some of the stopwords


def remove_punctuation_and_stopwords(sms):
   sms_no_punctuation = [ch for ch in sms if ch not in string.punctuation]
   sms_no_punctuation = "".join(sms_no_punctuation).split()

   sms_no_punctuation_no_stopwords = \
      [word.lower() for word in sms_no_punctuation if word.lower() not in stopwords.words("english")]

   return sms_no_punctuation_no_stopwords

data['text'].apply(remove_punctuation_and_stopwords).head()

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer = remove_punctuation_and_stopwords).fit(data['text'])
print(len(bow_transformer.vocabulary_))
# It will print total no of words we have in our vocalbury
# Applying vocalbury to all words , here vocalbury = bow-transformer
bow_data = bow_transformer.transform(data['text'])
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(bow_data)

# To check the importanct of words from a ham label sms .
sample_ham = data['text'][4]
bow_sample_ham = bow_transformer.transform([sample_ham])
tfidf_sample_ham = tfidf_transformer.transform(bow_sample_ham)
print(tfidf_sample_ham)
