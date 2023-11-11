from flask import Flask, render_template, request
from utils.vectorizer import process_and_encode_articles,encode_dataset
from utils.indexer import get_index, index_dataset,similarity_search

app = Flask(__name__)

@app.route('/')
def index():
    # Render the main page with the input form
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Get the text passage from the form
    text_passage = request.form['query']
    summary, embedding = process_and_encode_articles([text_passage])

    index = get_index()

    D, I = similarity_search(embedding[0].reshape(1,-1), 5, index)

    # Render a template with the results
    return render_template('results.html', results= I)

if __name__ == '__main__':
    app.run(debug=True)
