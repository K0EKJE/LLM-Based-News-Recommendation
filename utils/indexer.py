import os
import numpy as np
import faiss


def get_index():
        """
        Create and return a Faiss index based on embeddings stored in 'dataset/embeddings.npy'

        Parameters:
        None

        Returns:
        faiss.IndexFlatL2: a Faiss flat L2 index containing the loaded embeddings
        """

        # Check if the embeddings file exists
        if os.path.exists('dataset/embeddings.npy'):
            # Load the embeddings
            embeddings = np.load('dataset/embeddings.npy')
        else:
            print('No existing embeddings found')

        # Create a Faiss index (a simple flat L2 index)
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Add vectors to the index
        index.add(embeddings)

        return index

def similarity_search(vec, k, index):
        """
        Perform similarity search on a Faiss index

        Parameters:
        - vec (numpy.ndarray): the vector for which to search similar vectors
        - k (int): the number of nearest neighbors to retrieve
        - index (faiss.IndexFlatL2): the Faiss index to search

        Returns:
        tuple: a tuple containing two arrays, which are D (distances) and I (indices) of the k nearest neighbors
        """

        # Search the index (D is distance, I is index of neighbors)
        D, I = index.search(vec, k)
        return D, I

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
