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
    links=[]
    for r in ddgs_text_gen:
        titles.append(r['title'])
        body.append(r['body'])
        links.append(r['href'])
    raw_text = titles + body

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

    return important_keywords,links,raw_text

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

#===================================================================================================
import re
from nltk.corpus import stopwords

def find_related_words(content, keyword):
    dict={}
    for i in keyword:
        pattern = r'\b(\w{3,})\s+' + re.escape(i) + r'\s+(\w{3,})\b'
        matches = re.findall(pattern, content, re.IGNORECASE)
    
    # Extract words from the tuples and return as a list
        words = [word for match in matches for word in match]
        dict[i]=words
    
    return dict




#===================================================================================================
    
# Example usage

# Extract relevant n-gram keywords from Wikipedia content
query = "what is the capital of egypt"
keywords,links ,raw_text= get_google_results(query)
unique_keywords = list(set(get_top_keywords(query,keywords)))
query_tokens = word_tokenize(query.lower())
unique_keywords = [keyword for keyword in unique_keywords if keyword not in query_tokens]


random.shuffle(unique_keywords)
# print(unique_keywords)
raw_text = " ".join(raw_text)
related_words = find_related_words(raw_text, unique_keywords)
words=[]
for i,j in related_words.items():
    if len(j)>0:
        words.append(str(i)+" "+str(j[0]))
words=list(set(words))
def clear_words(words):
    stop_words = stopwords.words('english')
    for i in words.split():
        if i in stop_words:
            return False
        if len(i)<3:
            return False
    return True
words=[i for i in words if clear_words(i)]
unique_keywords=unique_keywords+words
random.shuffle(unique_keywords)
print(unique_keywords)



