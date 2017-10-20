from typing import List, Set
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity
from gensim.models import LsiModel
from nltk.stem.porter import PorterStemmer
import re
import random
import codecs
import string
random.seed(123)


class IR:
    def __init__(self, documents: List, stop_words: Set, stemming: bool = True):
        self.documents = documents
        self.stop_words = stop_words
        self.stemming = stemming
        self.proccess_documents(self.documents)
        self.dictionary = None
        self.corpus = None
        self.tfidf_model = None
        self.tfidf_corpus = None
        self.tfidf_similarity = None
        self.lsi_model = None
        self.lsi_corpus = None
        self.lsi_similarity = None
        self.build_dictionary()
        self.build_models()

    """ 1. Data loading and preprocessing """
    def proccess_documents(self, documents: List) -> List:
        # Tokenize documents
        result = self.tokenize(documents)
        # Lowercase all words
        result = list(map(lambda x: self.lowercase(x), result))
        # Remove stop words
        result = self.filter_stopwords(result)
        # Remove text punctuation
        result = self.remove_text_punctation(result)
        # Stem words
        if self.stemming:
            result = self.port_stem(result)
        # Remove empty words from all documents
        return self.filter_empty_words(result)

    """ 2. Dictionary building """
    def build_dictionary(self):
        documents = self.proccess_documents(self.documents)
        self.dictionary = Dictionary(documents)
        self.corpus = list(map(lambda doc: self.dictionary.doc2bow(doc), documents))

    """ 3. Retrieval Models """
    def build_models(self):
        # Create tfidf model
        self.tfidf_model = TfidfModel(self.corpus)

        # Map bag of words to (word-index, word-weight)
        self.tfidf_corpus = list(map(lambda c: self.tfidf_model[c], self.corpus))

        self.tfidf_similarity = MatrixSimilarity(self.tfidf_corpus)

        self.lsi_model = LsiModel(self.tfidf_corpus, id2word=self.dictionary, num_topics=100)

        self.lsi_corpus = list(map(lambda c: self.lsi_model[c], self.tfidf_corpus))

        self.lsi_similarity = MatrixSimilarity(self.lsi_corpus)

    def filter_stopwords(self, paragraphs: List) -> List:
        return list(map(lambda p: list(filter(lambda x: x not in self.stop_words, p)), paragraphs))

    """ 4. Querying  """
    def procces_query(self, query: str) -> List:
        tokenized = self.tokenize([query])
        lowered = list(map(lambda x: self.lowercase(x), tokenized))
        stop_word_filtered = self.filter_stopwords(lowered)
        punctation_filtered = self.remove_text_punctation(stop_word_filtered)
        if self.stemming:
            return self.port_stem(punctation_filtered)
        return punctation_filtered

    def tfidf_query(self, query: str, number_of_results: int = 3) -> None:
        # Proccess query
        proccessed_query = self.procces_query(query)
        query_corpus = self.dictionary.doc2bow(proccessed_query[0])
        query_tfidf = self.tfidf_model[query_corpus]
        similarity = enumerate(self.tfidf_similarity[query_tfidf])
        # Query most relevant paragraphs using TFIDF model
        query_result = sorted(similarity, key=lambda kv: -kv[1])[:number_of_results]
        # Print search result
        for result in query_result:
            number, _ = result
            print("Paragraph:", number)
            print(self.documents[number], "\n")

    def lsi_query(self, query: str, number_of_results: int = 3) -> None:
        # Proccess query
        proccessed_query = self.procces_query(query)
        query_corpus = self.dictionary.doc2bow(proccessed_query[0])
        tfidf_query = self.tfidf_model[query_corpus]
        lsi_query = self.lsi_model[tfidf_query]

        # Fetch most relevant topics
        relevant_topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:number_of_results]

        # Print
        for result in relevant_topics:
            number, _ = result
            print("Topic:", number)
            print(self.lsi_model.show_topic(number))
        print()

        # Find most relevant paragraphs using LSI similarity
        doc2similarity = enumerate(self.lsi_similarity[lsi_query])
        query_result = sorted(doc2similarity, key=lambda kv: -kv[1])[:number_of_results]
        # Print query result
        for result in query_result:
            number, _ = result
            print("Paragraph:", number)
            print(self.documents[number], "\n")

    """ All methods below is helpers to preproccess both documents and queries. """
    @staticmethod
    def filter_empty_words(paragraphs: List) -> List:
        return list(map(lambda p: list(filter(lambda w: w != "", p)), paragraphs))

    @staticmethod
    def tokenize(documents: List) -> List:
        return list(map(lambda x: x.split(), documents))

    @staticmethod
    def lowercase(words: List) -> List:
        return list(map(lambda s: s.lower(), words))

    @staticmethod
    def port_stem(documents: List) -> List:
        stemmer = PorterStemmer()
        return list(map(lambda p: list(map(lambda w: stemmer.stem(w), p)), documents))

    @staticmethod
    def remove_text_punctation(documents: List) -> List:
        regular = "[" + string.punctuation + "\n\r\t" + "]"
        return list(map(lambda p: list(map(lambda w: re.sub(regular, "", w), p)), documents))


def remove_paragraphs_containing(liste: List, word: str) -> List:
    return list(filter(lambda x: word not in x, liste))


def partition_book_to_paragraphs(filename: str) -> List:
    file = codecs.open(filename, "r", "utf-8")
    return file.read().split("\r\n\r")


# Partition into paragraphs
paragraphs = partition_book_to_paragraphs("pg3300.txt")

# Remove paragraphs containing "Gutenberg"
paragraphs = remove_paragraphs_containing(paragraphs, "Gutenberg")

# Create Set of stop words
stop_words = set(open("stopwords.txt", "r").read().split(","))

# Create model
model = IR(paragraphs, stop_words, True)


""" Report and try to interpret first 3 LSI topics """
print(model.lsi_model.show_topics()[:3])
"""
Topic 0: Firms
Topic 1: Real Estate
Topic 3: Gold/Silver/Coins 
"""


""" Report tdfidf weights """
query = model.procces_query("How taxes influence Economics?")[0]
query_corpus = model.dictionary.doc2bow(query)
print(model.tfidf_model[query_corpus])


""" Report top 3 the most relevant paragraphs """
model.tfidf_query("What is the function of money?", 3)


"""  Report top 3. topics with the most significant weights. """
model.lsi_query("How taxes influence Economics?", 3)

""" Compare retrieved paragraphs with the paragraphs found for TF-IDF model """
model.tfidf_query("How taxes influence Economics?", 3)
model.lsi_query("How taxes influence Economics?", 3)

"""  TD-IDF model and LSI results only have paragraph number 2063 in common """


