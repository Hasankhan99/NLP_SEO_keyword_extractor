"""
This code snippet demonstrates a keyword extraction process using Natural Language Processing (NLP) techniques.
It uses NLTK (Natural Language Toolkit) for tokenization, stop word removal, and n-gram generation.
It also utilizes the SentenceTransformer library for computing cosine similarity between the query and keywords.

Functions:
    - cosine_similarity(query, keywords): Computes the cosine similarity between the query and a list of keywords.
    - get_top_keywords(query, keywords): Retrieves the top keywords based on their cosine similarity scores.
    - extract_keywords(all_text, query, use_ngrams=True): Extracts important keywords from the given text based on the query.
    - get_data(query): Retrieves relevant text data based on the query.

Usage:
    1. Set the value of the 'query' variable to the desired search query.
    2. Call the 'get_data(query)' function to retrieve relevant text data from a source (e.g., DDGS - DuckDuckGo Instant Answer API).
    3. Call the 'extract_keywords(all_text, query, use_ngrams=True)' function to extract important keywords from the text.
       Set 'use_ngrams' to True if you want to include n-grams (word combinations) as keywords.
    4. Call the 'get_top_keywords(query, important_keywords)' function to obtain the top keywords based on cosine similarity scores.
    5. The resulting top keywords will be printed to the console.

Dependencies:
    - NLTK: Provides natural language processing functionalities. Install it using 'pip install nltk'.
    - SentenceTransformer: A library for computing sentence embeddings. Install it using 'pip install sentence-transformers'.
"""

from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np
from duckduckgo_search import DDGS


def cosine_similarity(query, keywords):
    """
    Computes the cosine similarity between the query and a list of keywords.

    Args:
        query (str): The query string.
        keywords (list): A list of keywords to compare with the query.

    Returns:
        dict: A dictionary mapping keywords to their cosine similarity scores.
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence = query.lower()
    sentence_embedding = model.encode([sentence])[0]
    keyword_embeddings = model.encode(keywords)
    similarities = np.dot(keyword_embeddings, sentence_embedding) / (np.linalg.norm(keyword_embeddings, axis=1) * np.linalg.norm(sentence_embedding))
    dic = {}
    for keyword, similarity in zip(keywords, similarities):
        dic[keyword] = similarity
    return dic


def get_top_keywords(query, keywords):
    """
    Retrieves the top keywords based on their cosine similarity scores.

    Args:
        query (str): The query string.
        keywords (list): A list of keywords.

    Returns:
        list: The top keywords based on their cosine similarity scores.
    """
    dic = cosine_similarity(query, keywords)
    top_keywords = sorted(dic, key=dic.get, reverse=True)[:50]
    return top_keywords


def extract_keywords(all_text, query, use_ngrams=True):
    """
    Extracts important keywords from the given text based on the query.

    Args:
        all_text (str): The concatenated text containing relevant information.
        query (str): The query string.
        use_ngrams (bool): Indicates whether to include n-grams (word combinations) as keywords. Default is True.

    Returns:
        list: A list of important keywords extracted from the text.
    """
    query_tokens = word_tokenize(query.lower())
    keywords = word_tokenize(all_text)
    keywords = [token for token in keywords if token.isalpha()]
    keywords = [token for token in keywords if token not in stopwords.words('english')]
    keywords = [token for token in keywords if len(token) > 2]
    if use_ngrams:
        n_gram_keywords = keywords + [' '.join(gram) for gram in ngrams(keywords, 2)]
        important_keywords = [token for token in n_gram_keywords if token not in query_tokens]
    else:
        important_keywords = [token for token in keywords if token not in query_tokens]
    important_keywords = list(set(important_keywords))
    return important_keywords


def get_data(query):
    """
    Retrieves relevant text data based on the query.

    Args:
        query (str): The query string.

    Returns:
        str: The concatenated text containing relevant information.
    """
    # DDGS: DuckDuckGo Instant Answer API
    ddgs = DDGS()

    ddgs_text_gen = ddgs.text(query, region='us-en', safesearch='on', timelimit='y')

    titles = []
    body = []
    for r in ddgs_text_gen:
        titles.append(r['title'])
        body.append(r['body'])

    def remove_stop_words(text):
        stop_words = stopwords.words('english')
        return ' '.join([word for word in text.split() if word not in stop_words])

    titles = [remove_stop_words(title) for title in titles]
    all_titles = "".join(titles)

    body = [remove_stop_words(text) for text in body]
    all_body = "".join(body)

    all_text = all_titles + all_body
    all_text = " ".join(list(set([i.lower() for i in all_text.split()])))
    return all_text


# Example usage
query = "how to convert int to string in php"
all_text = get_data(query)
important_keywords = extract_keywords(all_text, query)
top_keywords = get_top_keywords(query, important_keywords)
top_keywords = [keyword for keyword in top_keywords if keyword not in stopwords.words('english')]
print(top_keywords)
