# News Recommendation Web Application

A LLM-based News article recommendation web application. Receives URL or a text corpus as input, performs index searching with [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss) (FAISS), and returns hyperlinks of the most similar articles in the database. If URL is used as an input, the application will automatically run a scraper to scrape the articles.

All the articles, including the input, will be summarized with a [Bart model](https://huggingface.co/docs/transformers/model_doc/bart) by default (or a [T5 model](https://huggingface.co/docs/transformers/model_doc/t5)) before getting encoded by a [SentenceTransformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), which maps sentences and paragraphs to a 384-dimensional dense vector space. The index search algorithm will then compare the input vector and the database, returning the indices of the Top 5 results. The final output will be retrieved from our dataset based on the indices.

We currently host our example dataset on AWS S3, which has 9800+ entries, and it is publicly available. Instead of the whole dataset, we stored the partitions of it so that we only need to retrieve the partitions that contain the final indices. We have also defined functions to automate the process of dataset partitioning, uploading and reading from AWS S3.

## Project Structure:

<p align="center">
  <img src="Project structure.png" style="width:900px;height:500px;">
</p>

## Project Structure Tree:
```
News-Recommendation/
│
├── app/
│   ├── static/
│   │   └── styles.css            # CSS styles
│   ├── templates/
│   │   ├── index.html            # Main page template
│   │   └── results.html          # Results display template
│   │   └── error.html            # Error display template
│   ├── __init__.py               # Initialize Flask app
│   └── routes.py                 # Flask routes
│
├── dataset/
│   ├── partitioned_embeddings/   # Example vector dataset partitions
│   ├── partitioned_nyt/          # Example dataset partitions
│   └── embeddings.npy            # Encoded vectors of dataset
│
├── news_articles/                # Scrapy project files
│   ├── news_articles/
│   │   ├── spiders/
│   │   │   └── __init__.py
│   │   │   └── news_spider.py    # Spider class definition
│   │   └── __init__.py     
│   │   └── items.py
│   │   └── middlewares.py
│   │   └── pipelines.py
│   │   └── settings.py
│   │   └── spider_runner.py      # Spider runner definition
│   └── scrapy.cfg              
│
├── utils/
│   ├── vectorizer.py             # Script for article vectorization
│   └── indexer.py                # Script for creating and querying the index
│   └── util.py                   # Script for dataset partition and interaction with S3
│
├── requirements.txt              # Python dependencies
└── run.py                        # Entry point to run the Flask app
```
