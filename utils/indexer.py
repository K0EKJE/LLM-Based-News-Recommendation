import os
import numpy as np
import faiss



def get_index():
        # Check if the embeddings file exists
        if os.path.exists('dataset/embeddings.npy'):
            # Load the embeddings
            embeddings = np.load('dataset/embeddings.npy')
        else:
            print('No existing embeddings found')

        # Create a Faiss index - here we use a simple flat L2 index
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Add vectors to the index
        index.add(embeddings)
        
        return index

def index_dataset(df):
        # Check if the embeddings file exists
        if os.path.exists('dataset/embeddings.npy'):
            # Load the embeddings
            embeddings = np.load('dataset/embeddings.npy')
        else:
            # Call the function to encode the dataset
            embeddings = encode_dataset(df)
            np.save('dataset/embeddings.npy', embeddings)

        # Create a Faiss index - here we use a simple flat L2 index
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Add vectors to the index
        index.add(embeddings)
        
        return index
        
def similarity_search(vec, k, index):

        # Search the index
        D, I = index.search(vec, k)  # D is distance, I is index of neighbors
        return D, I