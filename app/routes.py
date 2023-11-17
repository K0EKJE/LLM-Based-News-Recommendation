from flask import Flask, render_template, request
from utils.vectorizer import process_and_encode_articles,encode_dataset, preprocess_text, download_parse_article
from utils.indexer import get_index, index_dataset,similarity_search
import pandas as pd
from flask import Flask, jsonify, session


app = Flask(__name__)

# app.secret_key = 'hahaha'
#
# @app.route('/progress')
# def progress():
#     # Example: Fetch progress from session
#     progress = session.get('progress', 0)
#     return jsonify({'progress': progress})

@app.route('/')
def index():
    # Render the main page with the input form
    return render_template('index.html')

@app.route('/search_paragraph', methods=['POST'])
def search_paragraph():
    # Get the text passage from the form
    text_passage = request.form['paragraph']
    if len(text_passage) > 5000:
        return render_template('error.html', message="Paragraph too long. Maximum 5000 characters allowed.")

    summary, embedding = process_and_encode_articles([text_passage])

    index = get_index()
    D, I = similarity_search(embedding[0].reshape(1,-1), 5, index)

    df = pd.read_csv('dataset/example_df.csv')
    # Render a template with the results
    result = {i:j for i,j in zip(df['headline'][I[0]],df['link'][I[0]])}

    return render_template('results.html', results= result)

@app.route('/search_url', methods=['POST'])
def search_url():

    raw_text = download_parse_article(request.form['url'])
    text_passage = preprocess_text(raw_text)

    summary, embedding = process_and_encode_articles([text_passage])

    index = get_index()
    D, I = similarity_search(embedding[0].reshape(1,-1), 5, index)

    df = pd.read_csv('dataset/example_df.csv')
    # Render a template with the results
    result = {i:j for i,j in zip(df['headline'][I[0]],df['link'][I[0]])}

    return render_template('results.html', results= result)

if __name__ == '__main__':
    app.run(debug=True)
