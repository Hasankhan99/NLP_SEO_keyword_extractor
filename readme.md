# Keyword Extraction with NLP

This Python script demonstrates keyword extraction using Natural Language Processing (NLP) techniques. It utilizes NLTK (Natural Language Toolkit) for tokenization, stop word removal, and n-gram generation. It also employs the SentenceTransformer library for computing cosine similarity between a query and keywords. The DuckDuckGo Instant Answer API is used to retrieve relevant text data.

## Features

- Extract important keywords from a given text based on a query.
- Compute cosine similarity scores between the query and keywords.
- Retrieve the top keywords based on their similarity scores.

## Dependencies

- NLTK: Install using `pip install nltk`.
- SentenceTransformer: Install using `pip install sentence-transformers`.
- duckduckgo_search: Install using `pip install duckduckgo-search` (assumed import: `from duckduckgo_search import DDGS`).

## Usage

1. Set the value of the `query` variable in the script to your desired search query.
2. Run the script.
3. The script will retrieve relevant text data using the DuckDuckGo Instant Answer API.
4. Important keywords will be extracted from the text based on the query.
5. The top keywords will be displayed in the console.

## How It Works

The script follows these steps:

1. Retrieves relevant text data based on the query using the DuckDuckGo Instant Answer API.
2. Processes the text data by removing stop words and concatenating the titles and bodies.
3. Extracts important keywords from the text based on the query, using tokenization, stop word removal, and n-gram generation.
4. Computes the cosine similarity between the query and the extracted keywords using SentenceTransformer.
5. Retrieves the top keywords based on their similarity scores.
6. Displays the top keywords in the console.

Feel free to modify the script according to your specific needs and integrate it into your own projects.

## Credits

The code utilizes the NLTK library for NLP functionalities, the SentenceTransformer library for sentence embeddings, and the duckduckgo_search library for retrieving text data.

