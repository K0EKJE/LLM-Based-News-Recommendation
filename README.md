# News-Recommendation

## Project Structure Tree:
```
News-Recommendation/
│
├── app/
│   ├── static/
│   │   └── styles.css          # CSS styles
│   ├── templates/
│   │   ├── index.html          # Main page template
│   │   └── results.html        # Results display template
│   │   └── error.html          # Error display template
│   ├── __init__.py             # Initialize Flask app
│   └── routes.py               # Flask routes
│
├── utils/
│   ├── vectorizer.py           # Script for article vectorization
│   └── indexer.py              # Script for creating and querying the index
│
├── dataset/
│   └── articles.csv            # Dataset of articles
│   └── embeddings.npy          # Encoded vectors of dataset
│
├── requirements.txt            # Python dependencies
└── run.py                      # Entry point to run the Flask app
```
