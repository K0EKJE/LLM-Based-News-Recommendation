import pandas as pd
import numpy as np
import json
from datetime import datetime

# Function to extract keys from a json string
def extract_key(json_string, key):
    
    try:
        # Replace single quotes with double quotes
        json_string = json_string.replace("'", "\"")
        
        # Add double quotes around None to make it a valid JSON value
        json_string = json_string.replace('None', 'null')
        
        # Load JSON string into a Python dictionary
        data_dict = json.loads(json_string)
        
        # Extract the content of the specified key
        return data_dict.get(key, None)
    
    except (json.JSONDecodeError, AttributeError):
        return None

# Function to clean raw dataset
def prepare_df(df):
    
    # Select useful columns
    df = df[['web_url', 'headline', 'section_name', 'subsection_name', 'abstract', 'byline', 'pub_date', 'text']]
    
    # Drop NA
    df = df.dropna(subset=['web_url', 'headline', 'section_name', 'abstract', 'byline', 'pub_date'])
    
    # Rename columns
    df = df.rename(columns={'web_url':'link', 'section_name':'category', 'subsection_name':'sub_category',
               'abstract':'short_description', 'byline':'authors', 'pub_date':'date'})
    
    # Extract headline
    df['headline'] = df['headline'].apply(lambda x: extract_key(x, 'main'))
    
    # Extract authors
    df['authors'] = df['authors'].apply(lambda x: extract_key(x, 'original'))
    
    # Replace empty strings with None in the 'authors' column
    df['authors'] = df['authors'].replace('', None)
    
    # Remove 'By ' from the 'authors' column
    df['authors'] = df['authors'].str.replace('By ', '')
    
    # Convert the 'date' column to datetime and extract the date component
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Remove rows with texts less than 50 characters
    df = df[(df['text'].apply(len)>=50) & (df['short_description'].apply(len)>=2)]
    
    # Remove rows with repeated texts
    df = df.drop_duplicates(subset='text', keep='first')
    
    return(df)

# Function to preprocess scraped texts
def preprocess_text(text):
    
    # Convert all characters to lowercase
    text = text.lower()
    
    # Remove all non-alphanumeric characters (except spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Reduce multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove stop words
    stop = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop])
    
    return(text)