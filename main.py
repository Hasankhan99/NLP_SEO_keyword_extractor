from nltk.util import ngrams
from duckduckgo_search import DDGS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random

def get_google_results(query):
    ddgs = DDGS()

    
    ddgs_text_gen = ddgs.text(query, region='wt-wt', safesearch='on', timelimit='y')

    titles=[]
    body=[]
    for r in ddgs_text_gen:
        titles.append(r['title'])
        body.append(r['body'])

    def remove_stop_words(text):
        stop_words = stopwords.words('english')
        return ' '.join([word for word in text.split() if word not in stop_words])

    titles = [remove_stop_words(title) for title in titles]
    all_titles="".join(titles)

    body = [remove_stop_words(text) for text in body]
    all_body="".join(body)

    all_text = all_titles + all_body
    all_text=" ".join(list(set([i.lower() for i in all_text.split()])))



    query_tokens = word_tokenize(query.lower())
    keywords=word_tokenize(all_text)
    keywords = [token for token in keywords if token.isalpha()]
    keywords = [token for token in keywords if token not in stopwords.words('english')]
    keywords = [token for token in keywords if len(token)>2]
    important_keywords = [token for token in keywords if token not in query_tokens]

    important_keywords = list(set(important_keywords))

    return important_keywords

def cosine_similarity(query,keywords):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence = query
    
    sentence_embedding = model.encode([sentence])[0]
    keyword_embeddings = model.encode(keywords)
    # Calculate cosine similarity
    similarities = np.dot(keyword_embeddings, sentence_embedding) / (np.linalg.norm(keyword_embeddings, axis=1) * np.linalg.norm(sentence_embedding))
    dic={}
    for keyword, similarity in zip(keywords, similarities):
        # print(f"Keyword: {keyword}\nSimilarity: {similarity}")
        dic[keyword]=similarity
    return dic
    


def get_top_keywords(query,keywords):
    dic = cosine_similarity(query,keywords)
    top_keywords = sorted(dic, key=dic.get, reverse=True)[:50]
    return top_keywords

import wikipediaapi

def get_wikipedia_content(query):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(query)
    return page.text
def get_ngram(query):
    query_tokens = word_tokenize(query.lower())
    query_tokens = [token for token in query_tokens if token.isalpha()]
    query_tokens = [token for token in query_tokens if token not in stopwords.words('english')]
    query_tokens = [token for token in query_tokens if len(token)>2]
    content=""
    for i in query_tokens:
        content+=get_wikipedia_content(i)
    if content=="":
        return []
    
    n_words_list=[]
    for i in range(2,4):
        n_words_list.extend(find_ngram_words(content,i))
    # n_words_list = [token for token in n_words_list if token.isalpha()]
    
    def clean_text(text):
        import re
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        stopwords_ = stopwords.words('english')
        text = ' '.join([word for word in text.split() if word not in stopwords_])
        return text


    n_words_list=[i for i in n_words_list if i not in query_tokens]
    n_words_list=[clean_text(i) for i in n_words_list]
    n_words_list=list(set(n_words_list))
    n_words_list=n_words_list[:100]

    n_keywords=get_top_keywords(query,n_words_list)
    n_keywords=[i for i in n_keywords if i not in query_tokens]
    def validate_keyword(keyword):
        if len(keyword.split())>1:
            for i in keyword.split():
                if i in query_tokens:
                    return False 
            return True
    n_keywords=[i for i in n_keywords if validate_keyword(i)]
    return n_keywords[:20]
    


    
def find_ngram_words(content, n):
    
    # Tokenize the content
    tokens = word_tokenize(content)
    
    # Generate n-grams
    ngram_list = list(ngrams(tokens, n))
    ngram_words = [' '.join(gram) for gram in ngram_list]
    return ngram_words
    
    
# Example usage

# Extract relevant n-gram keywords from Wikipedia content
query = "what is the capital of egypt"
keywords = get_google_results(query)
unique_keywords = list(set(get_top_keywords(query,keywords)))
query_tokens = word_tokenize(query.lower())
unique_keywords = [keyword for keyword in unique_keywords if keyword not in query_tokens]
unique_keywords=get_ngram(query)+unique_keywords

random.shuffle(unique_keywords)
print(unique_keywords)



# Find 2-gram words in the content

