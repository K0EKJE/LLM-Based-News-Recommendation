from newspaper import Article
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer

def download_parse_article(url):
        """
        Download and parse the text of an article from the given URL

        Parameters:
        - url (str): the URL of the article

        Returns:
        str: the text content of the article
        """

        # Create an Article object
        article = Article(url)

        # Download and parse the content
        article.download()
        article.parse()

        return article.text


def preprocess_text(text):
        """
        Preprocesse the given text

        Parameters:
        - text (str): the input text for preprocessing

        Returns:
        str: the preprocessed text
        """

        # Convert to lowercase, , , , and
        text = text.lower()

        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenization
        words = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        preprocessed_text = ' '.join(lemmatized_words)

        return preprocessed_text


def summarize_text(preprocessed_text, model_name='bart'):
        """
        Summarize the preprocessed text using the specified model (BART or T5)

        Parameters:
        - preprocessed_text (str): the preprocessed text to be summarized
        - model_name (str): the name of the summarization model ('bart' or 't5')

        Returns:
        str: the summarized text.
        """

        # Summarize the text using either BART or T5 model based on the specified model name
        if model_name.lower() == 'bart':
            # Initialize BART tokenizer and model
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

            # Tokenize and encode the preprocessed text for BART
            inputs = tokenizer.encode("summarize: " + preprocessed_text, return_tensors='pt', max_length=1024, truncation=True)

            # Generate a summary using BART model
            summary_ids = model.generate(inputs, max_length=150, min_length=40,
                                         length_penalty=2.0, num_beams=4, early_stopping=True)

        elif model_name.lower() == 't5':
            # Initialize T5 tokenizer and model
            tokenizer = T5Tokenizer.from_pretrained('t5-large')
            model = T5ForConditionalGeneration.from_pretrained('t5-large')

            # Tokenize and encode the preprocessed text for T5
            input_ids = tokenizer("summarize: " + preprocessed_text, return_tensors="pt",
                                  max_length=512, truncation=True).input_ids

            # Generate a summary using T5 model
            summary_ids = model.generate(input_ids, max_length=150, min_length=40,
                                         length_penalty=2.0, num_beams=4, early_stopping=True)

        else:
            # Raise an error for an invalid model name
            raise ValueError("Model name should be 'bart' or 't5'")

        # Decode the generated summary and skip special tokens
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary




def process_and_encode_articles(texts, model_name='bart'):
        """
        Process and encode a list of articles' texts using the specified model ('bart' or 't5')

        Parameters:
        - texts (list): a list of article texts
        - model_name (str): the name of the summarization model ('bart' or 't5')

        Returns:
        tuple: two lists, which are summaries and embeddings
        """

        # Check if input is a list
        if isinstance(texts, str):
            raise TypeError("Input cannot be str")

        summaries = []
        embeddings = []

        # Use tqdm for the progress bar
        for i, text in tqdm(enumerate(texts), total=len(texts), desc="Processing Articles"):
            # Preprocess, summarize, and encode each article
            preprocessed_text = preprocess_text(text)
            summary = summarize_text(preprocessed_text, model_name)
            embedding = encode_text(summary)
            summaries.append(summary)
            embeddings.append(embedding)

        return summaries, embeddings



# Function to process and encode articles from multiple URLs
def process_and_encode_url(urls, model_name='bart'):
        """
        Process and encode text from url input using the specified model ('bart' or 't5')

        Parameters:
        - texts (list): a list of urls
        - model_name (str): the name of the summarization model ('bart' or 't5')

        Returns:
        tuple: two lists, which are summaries and embeddings
        """
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



def encode_dataset(df, output_path):
        """
        Encode a dataset of articles from the 'text' column and saves the embeddings to a numpy file

        Parameters:
        - df (pandas.DataFrame): the dataset containing a 'text' column with article texts
        - output_path (str): the path to save the embeddings numpy file

        Returns:
        None
        """

        # Process and encode articles from the 'text' column of the DataFrame
        summaries, embedding = process_and_encode_articles(df['text'])

        # Save the embeddings to the specified output path as a numpy file
        np.save(output_path, embedding)

def process_single_article(text, model_name='bart'):
        preprocessed_text = preprocess_text(text)
        summary = summarize_text(preprocessed_text, model_name)
        embedding = encode_text(summary)
        return summary, embedding
