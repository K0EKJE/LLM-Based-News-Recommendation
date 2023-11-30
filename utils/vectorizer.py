from newspaper import Article
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np


from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer

# Function to download and parse an article
def download_parse_article(url):
        article = Article(url)
        article.download()
        article.parse()
        return article.text

# Function to preprocess text
def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        preprocessed_text = ' '.join(lemmatized_words)
        return preprocessed_text

# Function to encode text using Sentence Transformers
def encode_text(summary):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(summary)
        return embedding

# Function to summarize text using the specified model (BART or T5)
def summarize_text(preprocessed_text, model_name='bart'):
        if model_name.lower() == 'bart':
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
            inputs = tokenizer.encode("summarize: " + preprocessed_text, return_tensors='pt', max_length=1024, truncation=True)
            summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        elif model_name.lower() == 't5':
            tokenizer = T5Tokenizer.from_pretrained('t5-large')
            model = T5ForConditionalGeneration.from_pretrained('t5-large')
            input_ids = tokenizer("summarize: " + preprocessed_text, return_tensors="pt", max_length=512, truncation=True).input_ids
            summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        else:
            raise ValueError("Model name should be 'bart' or 't5'")

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

from tqdm import tqdm

# Function to process and encode articles from texts
def process_and_encode_articles(texts, model_name='bart'):
        if isinstance(texts, str):
            raise TypeError("Input cannot be str")

        summaries = []
        embeddings = []

        # Use tqdm for the progress bar
        for i, text in tqdm(enumerate(texts), total=len(texts), desc="Processing Articles"):
            preprocessed_text = preprocess_text(text)
            summary = summarize_text(preprocessed_text, model_name)
            embedding = encode_text(summary)
            summaries.append(summary)
            embeddings.append(embedding)

        return summaries, embeddings


    
# Function to process and encode articles from multiple URLs
def process_and_encode_url(urls, model_name='bart'):
        if isinstance(texts, str):
            raise TypeError("Input can not be str")

        summaries = []
        embeddings = []

        for i,url in enumerate(urls):
            print(f'Start embedding...{i}')
            text = download_parse_article(url)
            preprocessed_text = preprocess_text(text)
            summary = summarize_text(preprocessed_text, model_name)
            embedding = encode_text(summary)
            summaries.append(summary)
            embeddings.append(embedding)

        return summaries, embeddings
    

def encode_dataset(df):

        summaries, embedding = process_and_encode_articles(df['text'])
        np.save('dataset/embeddings.npy', embedding)
 
        # To load the embeddings back
        # loaded_embeddings = np.load('dataset/embeddings.npy')

    
def process_single_article(text, model_name='bart'):
        preprocessed_text = preprocess_text(text)
        summary = summarize_text(preprocessed_text, model_name)
        embedding = encode_text(summary)
        return summary, embedding


