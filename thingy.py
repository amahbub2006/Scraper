import pandas as pd
import numpy as np
import math
from collections import Counter

# Load the CSV files
movies_df = pd.read_csv(r'C:\Users\azfar\Downloads\archive (3)\tmdb_5000_movies.csv', low_memory=False)
credits_df = pd.read_csv(r'C:\Users\azfar\Downloads\archive (3)\tmdb_5000_credits.csv', low_memory=False)

# Convert 'id' columns to int64 for both DataFrames
movies_df['movie_id'] = pd.to_numeric(movies_df['movie_id'], errors='coerce').astype('Int64')
credits_df['movie_id'] = pd.to_numeric(credits_df['movie_id'], errors='coerce').astype('Int64')

# Merge the DataFrames on the 'id' column
merged_df = pd.merge(movies_df, credits_df, on='movie_id')

# Function to clean and prepare data
def clean_data(x):
    if isinstance(x, str):
        return ' '.join(word.lower().replace(" ", "") for word in x.split())
    return ''

# Combine relevant features into a single string
def create_soup(features):
    return ' '.join(features)

# List of features to use; adjust based on actual column names
features = ['keywords', 'cast', 'genres']  # Ensure these columns exist

# Handle missing columns
for feature in features:
    if feature in merged_df.columns:
        merged_df[feature] = merged_df[feature].apply(clean_data)
    else:
        print(f"Warning: Column '{feature}' is missing from the dataset")

# Create the "soup"
merged_df['soup'] = merged_df.apply(lambda x: create_soup([x[feature] for feature in features if feature in x]), axis=1)

# Function to compute TF-IDF
def compute_tf(word_dict, doc):
    tf_dict = {}
    doc_count = len(doc)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(doc_count)
    return tf_dict

def compute_idf(doc_list):
    idf_dict = {}
    n = len(doc_list)

    for doc in doc_list:
        if isinstance(doc, dict):
            for word in doc.keys():
                if word not in idf_dict:
                    idf_dict[word] = 1
                else:
                    idf_dict[word] += 1

    for word, val in idf_dict.items():
        idf_dict[word] = math.log(n / float(val))

    return idf_dict

def compute_tfidf(tf_dict, idf_dict):
    tfidf = {}
    for word, val in tf_dict.items():
        if word in idf_dict:
            tfidf[word] = val * idf_dict[word]
    return tfidf

# Function to get word count for each movie
def word_count(soup):
    if isinstance(soup, str):
        return Counter(soup.split())
    return Counter()

# Calculate TF-IDF matrix manually
docs = merged_df['soup'].apply(word_count)
# Create a vocabulary (set of all unique words)
all_words = set(word for doc in docs for word in doc.keys())
vocabulary = sorted(all_words)

# Function to create a TF-IDF vector for a given document
def create_tfidf_vector(doc_word_count, idf_dict):
    tf_dict = compute_tf(doc_word_count, doc_word_count)
    tfidf_dict = compute_tfidf(tf_dict, idf_dict)
    return np.array([tfidf_dict.get(word, 0) for word in vocabulary])

# Compute IDF values
idf = compute_idf(docs)

# Compute TF-IDF vectors
tfidf_vectors = docs.apply(lambda x: create_tfidf_vector(x, idf))

# Reset the index and create a reverse mapping of movie titles to indices
merged_df = merged_df.reset_index()
print("Columns in merged_df after resetting index:", merged_df.columns)

# Ensure the 'title_x_x' column exists
if 'title_x_x' in merged_df.columns:
    indices = pd.Series(merged_df.index, index=merged_df['title_x_x']).drop_duplicates()
else:
    raise KeyError("Column 'title_x_x' not found in merged_df.")

# Function to compute cosine similarity
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

# Function to get movie recommendations
def get_recommendations(title, tfidf_vectors=tfidf_vectors):
    if title not in indices:
        return "Movie not found in the dataset."

    idx = indices[title]
    cosine_sim = []

    for i, vec in enumerate(tfidf_vectors):
        if i != idx:
            sim = cosine_similarity(tfidf_vectors[idx], vec)
            cosine_sim.append((i, sim))
        else:
            cosine_sim.append((i, -1))  # Avoid self-comparison

    # Sort the movies based on the similarity scores
    sim_scores = sorted(cosine_sim, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[:10]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return merged_df['title_x_x'].iloc[movie_indices]

# Test the recommendation system
print(get_recommendations('The Godfather: Part III'))
