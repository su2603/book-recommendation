# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import warnings
import math
from wordcloud import WordCloud
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Ignore warnings
warnings.filterwarnings('ignore')

class BookRecommender:
    """Main class for book recommendation system"""
    
    def __init__(self):
        """Initialize the recommender system"""
        self.models = {}
        self.metrics = {}
        self.config = {
            'popularity_threshold': 50,
            'user_interactions_threshold': 100,
            'rating_threshold': 3,
            'book_ratings_threshold_perc': 0.1,
            'svd_factors': 15
        }
        
        # Load and preprocess data directly
        self.load_and_preprocess_data()
            
    def load_and_preprocess_data(self):
        """Load and preprocess the datasets"""
        start_time = time.time()
        print("Loading datasets...")

        # Specify dtypes to reduce memory usage
        users_dtypes = {
        'User-ID': np.int32,
        'Age': np.float32  # Using float32 since we'll have NaN values
        }
    
        books_dtypes = {
        'Year-Of-Publication': 'str',  # Will convert to numeric later
        'Book-Author': 'str', 
        'Publisher': 'str'
        }
    
        ratings_dtypes = {
        'User-ID': np.int32,
        'ISBN': 'str',
        'Book-Rating': np.int8  # Ratings are small integers
        }
    
        # Use chunking if the files are very large
        self.users = pd.read_csv('Users.csv', dtype=users_dtypes)
        self.books = pd.read_csv('Books.csv', dtype=books_dtypes)
        
        # For ratings, which is likely the largest file, consider chunking
        chunk_size = 500000  # Adjust based on your system's memory
        ratings_chunks = []
    
        for chunk in pd.read_csv('Ratings.csv', dtype=ratings_dtypes, chunksize=chunk_size):
            # Perform filtering on each chunk
            chunk = chunk[chunk['ISBN'].isin(self.books['ISBN'])]
            chunk = chunk[chunk['User-ID'].isin(self.users['User-ID'])]
            ratings_chunks.append(chunk)
    
        self.ratings = pd.concat(ratings_chunks)
        del ratings_chunks  # Free up memory
    
        print(f"Datasets loaded in {time.time() - start_time:.2f} seconds")
        print(f"Book_df shape is {self.books.shape}")
        print(f"Ratings_df shape is {self.ratings.shape}")
        print(f"Users_df shape is {self.users.shape}")
    
        # Preprocess data
        self._preprocess_users()
        self._preprocess_books()
        self._preprocess_ratings()

def _preprocess_books(self):
    """Preprocess books dataset with memory optimization"""
    print("Preprocessing books dataset...")
    
    # Drop image URL columns immediately to save memory
    self.books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
    
    # Clean Year-Of-Publication data
    self.books['Year-Of-Publication'] = pd.to_numeric(self.books['Year-Of-Publication'], errors='coerce')
    
    # Handle invalid years
    self.books.loc[(self.books['Year-Of-Publication'] > 2006) | (self.books['Year-Of-Publication'] == 0), 'Year-Of-Publication'] = np.nan
    self.books['Year-Of-Publication'].fillna(round(self.books['Year-Of-Publication'].median()), inplace=True)
    self.books['Year-Of-Publication'] = self.books['Year-Of-Publication'].astype(np.int16)  # Save memory with smaller int type
    
    # Fix specific incorrect entries
    corrections = {
        '0789466953': {
            'Year-Of-Publication': 2000,
            'Book-Author': "James Buckley",
            'Publisher': "DK Publishing Inc",
            'Book-Title': "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
        },
        '078946697X': {
            'Year-Of-Publication': 2000,
            'Book-Author': "Michael Teitelbaum",
            'Publisher': "DK Publishing Inc",
            'Book-Title': "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
        },
        '2070426769': {
            'Year-Of-Publication': 2003,
            'Book-Author': "Jean-Marie Gustave Le ClÃ?Â©zio",
            'Publisher': "Gallimard",
            'Book-Title': "Peuple du ciel, suivi de 'Les Bergers"
        }
    }
    
    for isbn, corrections_dict in corrections.items():
        for column, value in corrections_dict.items():
            self.books.loc[self.books.ISBN == isbn, column] = value
    
    # Fill NaN values in Publisher and Book-Author
    self.books.Publisher.fillna('other', inplace=True)
    self.books['Book-Author'].fillna('other', inplace=True)
    
    # Instead of creating a full concatenated content column for all books at once,
    # we'll create a function to generate it on-demand
    def get_content(row):
        return f"{row['Book-Title']} {row['Book-Author']} {row['Publisher']}"
    
    # This will be called only when needed in the content-based model
    self.get_book_content = get_content
        
        
    def _preprocess_users(self):
        """Preprocess users dataset"""
        print("Preprocessing users dataset...")
        
        # Clean Age column
        # Replace extreme ages with NaN
        self.users.loc[(self.users.Age > 100) | (self.users.Age < 5), 'Age'] = np.nan
        
        # Create Country column from Location
        self.users['Country'] = self.users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')
        self.users.drop('Location', axis=1, inplace=True)
        self.users['Country'] = self.users['Country'].astype('str')
        
        # Clean Country data
        self.users['Country'].replace([
            '','01776','02458','19104','23232','30064','85021','87510','alachua',
            'america','austria','autralia','cananda','geermany','italia',
            'united kindgonm','united sates','united staes','united state','united states','us'
        ], [
            'other','usa','usa','usa','usa','usa','usa','usa','usa',
            'usa','australia','australia','canada','germany','italy',
            'united kingdom','usa','usa','usa','usa','usa'
        ], inplace=True)
        
        # Fill NaN values with median by country
        self.users['Age'] = self.users['Age'].fillna(self.users.groupby('Country')['Age'].transform('median'))
        
        # Fill remaining NaN with mean
        self.users['Age'].fillna(self.users.Age.mean(), inplace=True)
        
    def _preprocess_books(self):
        """Preprocess books dataset"""
        print("Preprocessing books dataset...")
        
        # Clean Year-Of-Publication data
        self.books['Year-Of-Publication'] = self.books['Year-Of-Publication'].astype('str')
        
        # Fix specific incorrect entries
        corrections = {
            '0789466953': {
                'Year-Of-Publication': '2000',
                'Book-Author': "James Buckley",
                'Publisher': "DK Publishing Inc",
                'Book-Title': "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
            },
            '078946697X': {
                'Year-Of-Publication': '2000',
                'Book-Author': "Michael Teitelbaum",
                'Publisher': "DK Publishing Inc",
                'Book-Title': "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
            },
            '2070426769': {
                'Year-Of-Publication': '2003',
                'Book-Author': "Jean-Marie Gustave Le ClÃ?Â©zio",
                'Publisher': "Gallimard",
                'Book-Title': "Peuple du ciel, suivi de 'Les Bergers"
            }
        }
        
        for isbn, corrections_dict in corrections.items():
            for column, value in corrections_dict.items():
                self.books.loc[self.books.ISBN == isbn, column] = value
        
        # Convert Year-Of-Publication to numeric
        self.books['Year-Of-Publication'] = pd.to_numeric(self.books['Year-Of-Publication'], errors='coerce')
        
        # Handle invalid years
        self.books.loc[(self.books['Year-Of-Publication'] > 2006) | (self.books['Year-Of-Publication'] == 0), 'Year-Of-Publication'] = np.nan
        self.books['Year-Of-Publication'].fillna(round(self.books['Year-Of-Publication'].median()), inplace=True)
        
        # Drop image URL columns
        self.books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
        
        # Fill NaN values in Publisher and Book-Author
        self.books.Publisher.fillna('other', inplace=True)
        self.books['Book-Author'].fillna('other', inplace=True)
        
        # Create a concatenated genre field (for potential content-based filtering)
        self.books['content'] = self.books['Book-Title'] + ' ' + self.books['Book-Author'] + ' ' + self.books['Publisher']
        
    def _preprocess_ratings(self):
        """Preprocess ratings dataset"""
        print("Preprocessing ratings dataset...")
        
        # Keep only ratings for books that exist in books dataset
        self.ratings_new = self.ratings[self.ratings.ISBN.isin(self.books.ISBN)]
        print(f"Shape of dataset after filtering for existing books: {self.ratings_new.shape}")
        
        # Keep only ratings from users who exist in users dataset
        self.ratings_new = self.ratings_new[self.ratings_new['User-ID'].isin(self.users['User-ID'])]
        print(f"Shape of dataset after ensuring users exist: {self.ratings_new.shape}")
        
        # Separate implicit and explicit ratings
        self.ratings_explicit = self.ratings_new[self.ratings_new['Book-Rating'] != 0]
        self.ratings_implicit = self.ratings_new[self.ratings_new['Book-Rating'] == 0]
        print(f'Explicit ratings dataset shape: {self.ratings_explicit.shape}')
        print(f'Implicit ratings dataset shape: {self.ratings_implicit.shape}')
        
        # Create aggregate columns for ratings
        self.ratings_explicit['Avg_Rating'] = self.ratings_explicit.groupby('ISBN')['Book-Rating'].transform('mean')
        self.ratings_explicit['Total_No_Of_Users_Rated'] = self.ratings_explicit.groupby('ISBN')['Book-Rating'].transform('count')
        
        # Merge all datasets for the final dataset
        self.final_dataset = self.users.copy()
        self.final_dataset = pd.merge(self.final_dataset, self.ratings_explicit, on='User-ID')
        self.final_dataset = pd.merge(self.final_dataset, self.books, on='ISBN')
    
    # Helper methods for reducing redundancy
    def ensure_model(self, model_type):
        """Ensure a model is built before using it"""
        model_builders = {
            'popularity': self.build_popularity_model,
            'item_cf': self.build_item_based_cf,
            'svd': self.build_svd_model,
            'content_based': self.build_content_based_model
        }
        
        if model_type not in self.models and model_type in model_builders:
            model_builders[model_type]()
        
        return model_type in self.models

    def handle_not_found(self, entity_type, entity_id):
        """Create standardized response for not found entities"""
        return pd.DataFrame(
            {
                'ISBN': ['Not found'],
                'Book-Title': [
                    f'{entity_type} with {entity_type.lower()[:4]} {entity_id} not found in dataset'
                ],
            }
        )
        
    def build_popularity_model(self):
        """Build popularity-based filtering model"""
        print("Building popularity-based model...")
        
        # Calculate average rating and minimum user count threshold
        C = self.final_dataset['Avg_Rating'].mean()
        m = self.final_dataset['Total_No_Of_Users_Rated'].quantile(0.90)
        
        # Filter for books with enough ratings
        top_books = self.final_dataset.loc[self.final_dataset['Total_No_Of_Users_Rated'] >= m].copy()
        print(f"Using minimum threshold of {m} ratings per book. Dataset shape: {top_books.shape}")
        
        # Define weighted rating function
        def weighted_rating(x, m=m, C=C):
            v = x['Total_No_Of_Users_Rated']
            R = x['Avg_Rating']
            return (v/(v+m) * R) + (m/(m+v) * C)
        
        # Calculate scores and sort
        top_books['Score'] = top_books.apply(weighted_rating, axis=1)
        top_books = top_books.sort_values('Score', ascending=False)
        
        # Keep unique ISBNs only
        self.popular_books = top_books.drop_duplicates('ISBN').sort_values('Score', ascending=False)
        
        # Save model
        self.models['popularity'] = {
            'model': self.popular_books,
            'C': C,
            'm': m
        }
        
        return self.popular_books.head(20)
        
    def build_item_based_cf(self):
        """Build item-based collaborative filtering model"""
        print("Building item-based collaborative filtering model...")
        
        # Rename columns for consistency
        df_ratings = self.ratings_explicit.rename(columns={'User-ID':'userID', 'ISBN':'ISBN', 'Book-Rating':'bookRating'})
        
        # Filter users with at least n ratings
        user_ratings_threshold = self.config['rating_threshold']
        filter_users = df_ratings['userID'].value_counts()
        filter_users_list = filter_users[filter_users >= user_ratings_threshold].index.to_list()
        df_ratings_top = df_ratings[df_ratings['userID'].isin(filter_users_list)]
        
        # Filter top n% most rated books
        book_ratings_threshold_perc = self.config['book_ratings_threshold_perc']
        book_ratings_threshold = len(df_ratings_top['ISBN'].unique()) * book_ratings_threshold_perc
        filter_books_list = df_ratings_top['ISBN'].value_counts().head(int(book_ratings_threshold)).index.to_list()
        df_ratings_top = df_ratings_top[df_ratings_top['ISBN'].isin(filter_books_list)]
        
        # Generate ratings matrix
        ratings_matrix = df_ratings_top.pivot(index='userID', columns='ISBN', values='bookRating')
        ratings_matrix.fillna(0, inplace=True)
        ratings_matrix = ratings_matrix.astype(np.int32)
        
        # Calculate sparsity
        sparsity = 1.0 - len(df_ratings_top)/float(ratings_matrix.shape[0] * ratings_matrix.shape[1])
        print(f'The sparsity level of dataset is {sparsity*100:.2f}%')
        
        # Build KNN model
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        ratings_matrix_csr = csr_matrix(ratings_matrix.values)
        model_knn.fit(ratings_matrix_csr)
        
        # Save model
        self.models['item_cf'] = {
            'model': model_knn,
            'ratings_matrix': ratings_matrix,
            'books_data': self.books
        }
        
        return model_knn
    
    def build_svd_model(self):
        """Build SVD-based collaborative filtering model"""
        print("Building SVD-based collaborative filtering model...")
        
        # Rename columns for consistency
        df_ratings = self.ratings_explicit.rename(columns={'User-ID':'userID', 'ISBN':'isbn', 'Book-Rating':'book_rating'})
        
        # Filter users with many interactions
        users_interactions_count_df = df_ratings.groupby(['isbn', 'userID']).size().groupby('userID').size()
        users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= self.config['user_interactions_threshold']].reset_index()
        
        if len(users_with_enough_interactions_df) == 0:
            print("Warning: No users with enough interactions found. Lowering threshold.")
            # Lower the threshold if needed
            self.config['user_interactions_threshold'] = 50
            users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= self.config['user_interactions_threshold']].reset_index()
        
        interactions_from_selected_users_df = df_ratings.merge(users_with_enough_interactions_df, 
                                                             how='right',
                                                             left_on='userID',
                                                             right_on='userID')
        
        # Apply log transformation to user preferences
        def smooth_user_preference(x):
            return math.log(1+x, 2)
            
        interactions_full_df = interactions_from_selected_users_df.groupby(['isbn', 'userID'])['book_rating'].sum().apply(smooth_user_preference).reset_index()
        
        # Split data into train and test
        interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                    stratify=interactions_full_df['userID'], 
                                                                    test_size=0.20,
                                                                    random_state=42)
        
        # Create pivot table
        users_items_pivot_matrix_df = interactions_train_df.pivot(index='userID', 
                                                                columns='isbn', 
                                                                values='book_rating').fillna(0)
        
        users_items_pivot_matrix = users_items_pivot_matrix_df.values
        users_ids = list(users_items_pivot_matrix_df.index)
        
        # SVD decomposition
        NUMBER_OF_FACTORS_MF = self.config['svd_factors']
        U, sigma, Vt = svds(users_items_pivot_matrix, k=NUMBER_OF_FACTORS_MF)
        sigma = np.diag(sigma)
        
        # Reconstruct the prediction matrix
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
        cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns=users_items_pivot_matrix_df.columns, index=users_ids).transpose()
        
        # Create indexed dataframes for recommendation
        interactions_full_indexed_df = interactions_full_df.set_index('userID')
        interactions_train_indexed_df = interactions_train_df.set_index('userID')
        interactions_test_indexed_df = interactions_test_df.set_index('userID')
        
        # Save model components
        self.models['svd'] = {
            'predictions_df': cf_preds_df,
            'interactions_full_indexed_df': interactions_full_indexed_df,
            'interactions_train_indexed_df': interactions_train_indexed_df,
            'interactions_test_indexed_df': interactions_test_indexed_df,
            'books_data': self.books
        }
        
        return cf_preds_df
    
    def build_content_based_model(self):
        """Build content-based filtering model"""
        print("Building content-based filtering model...")
        
        # Create TF-IDF vectorizer for content
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.books['content'])
        
        # Compute cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Reset index of books
        self.books = self.books.reset_index()
        
        # Create a Series to map indices to book ISBNs
        indices = pd.Series(self.books.index, index=self.books['ISBN'])
        
        # Save model
        self.models['content_based'] = {
            'cosine_sim': cosine_sim,
            'indices': indices,
            'books_data': self.books
        }
        
        return cosine_sim
    
    # Refactored recommendation functions
    def get_recommendations_popularity(self, n=10):
        """Get recommendations using popularity-based filtering"""
        if not self.ensure_model('popularity'):
            return self.handle_not_found("Model", "popularity")
                
        return self.models['popularity']['model'].head(n)
    
    def get_recommendations_item_cf(self, book_isbn, n=10):
        """Get recommendations using item-based collaborative filtering"""
        if not self.ensure_model('item_cf'):
            return self.handle_not_found("Model", "item_cf")
                
        model_knn = self.models['item_cf']['model']
        ratings_matrix = self.models['item_cf']['ratings_matrix']
        
        if book_isbn not in ratings_matrix.columns:
            return self.handle_not_found("Book", book_isbn)
            
        book_idx = ratings_matrix.columns.get_loc(book_isbn)
        book_vector = ratings_matrix.iloc[:, book_idx].values.reshape(1, -1)
        book_vector = csr_matrix(book_vector)
        
        distances, indices = model_knn.kneighbors(book_vector, n_neighbors=n+1)
        
        similar_isbn_indices = [(ratings_matrix.columns[indices.flatten()[i]], distances.flatten()[i]) 
                             for i in range(len(distances.flatten())) if i != 0]
        
        similar_isbn = pd.DataFrame(similar_isbn_indices, columns=['ISBN', 'Similarity Score'])
        
        # Get book details
        return similar_isbn.merge(self.books, on='ISBN', how='left')
    
    def get_recommendations_svd(self, user_id, n=10):
        """Get recommendations using SVD"""
        if not self.ensure_model('svd'):
            return self.handle_not_found("Model", "svd")
                
        cf_preds_df = self.models['svd']['predictions_df']
        interactions_full_indexed_df = self.models['svd']['interactions_full_indexed_df']
        
        if user_id not in cf_preds_df.columns:
            return self.handle_not_found("User", user_id)
        
        # Get items user has already interacted with
        def get_items_interacted(user_id, interactions_df):
            interacted_items = interactions_df.loc[user_id]['isbn']
            return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
        
        try:
            items_to_ignore = get_items_interacted(user_id, interactions_full_indexed_df)
        except KeyError:
            items_to_ignore = set()
        
        # Sort predictions for the user
        sorted_user_predictions = cf_preds_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})
        
        # Filter out items user has already interacted with
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['isbn'].isin(items_to_ignore)].head(n)
        
        # Add book details
        recommendations_df = recommendations_df.merge(self.books, left_on='isbn', right_on='ISBN')
        
        return recommendations_df
    
    def get_recommendations_content_based(self, book_isbn, n=10):
        """Get recommendations using content-based filtering"""
        if not self.ensure_model('content_based'):
            return self.handle_not_found("Model", "content_based")
                
        cosine_sim = self.models['content_based']['cosine_sim']
        indices = self.models['content_based']['indices']
        books_data = self.models['content_based']['books_data']
        
        if book_isbn not in indices:
            return self.handle_not_found("Book", book_isbn)
        
        # Get index of the book
        idx = indices[book_isbn]
        
        # Get similarity scores for all books
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort based on similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n similar books (excluding itself)
        sim_scores = sim_scores[1:n+1]
        
        # Get book indices
        book_indices = [i[0] for i in sim_scores]
        similarity = [i[1] for i in sim_scores]
        
        # Create recommendations dataframe
        recommendations = books_data.iloc[book_indices]
        recommendations['Similarity Score'] = similarity
        
        return recommendations
    
    def hybrid_recommendations(self, user_id, book_isbn=None, n=10):
        """Get recommendations using hybrid approach"""
        # Get recommendations from different models
        if book_isbn:
            content_recs = self.get_recommendations_content_based(book_isbn, n=n)
            item_cf_recs = self.get_recommendations_item_cf(book_isbn, n=n)
        else:
            content_recs = pd.DataFrame()
            item_cf_recs = pd.DataFrame()

        svd_recs = self.get_recommendations_svd(user_id, n=n)
        popularity_recs = self.get_recommendations_popularity(n=n)

        # Combine all recommendations
        all_recs = pd.DataFrame()

        if (
            not content_recs.empty
            and 'ISBN' in content_recs.columns
            and content_recs['ISBN'].iloc[0] != 'Not found'
        ):
            content_recs['source'] = 'Content-Based'
            all_recs = pd.concat([all_recs, content_recs])

        if (
            not item_cf_recs.empty
            and 'ISBN' in item_cf_recs.columns
            and item_cf_recs['ISBN'].iloc[0] != 'Not found'
        ):
            item_cf_recs['source'] = 'Item-Based CF'
            all_recs = pd.concat([all_recs, item_cf_recs])

        if (
            not svd_recs.empty
            and 'ISBN' in svd_recs.columns
            and svd_recs['ISBN'].iloc[0] != 'Not found'
        ):
            svd_recs['source'] = 'SVD-Based CF'
            all_recs = pd.concat([all_recs, svd_recs])

        popularity_recs['source'] = 'Popularity-Based'
        all_recs = pd.concat([all_recs, popularity_recs])

        # Remove duplicates, prioritizing recommendations from more personalized methods
        priority_order = {'Content-Based': 1, 'Item-Based CF': 2, 'SVD-Based CF': 3, 'Popularity-Based': 4}
        all_recs['priority'] = all_recs['source'].map(priority_order)
        all_recs = all_recs.sort_values('priority').drop_duplicates('ISBN').head(n)

        return all_recs
    
    def evaluate_models(self, k=5):
        """Evaluate all recommendation models using k-fold cross-validation"""
        print("Evaluating recommendation models...")
        
        if not self.ensure_model('svd'):
            print("SVD model not available, building it first.")
            self.build_svd_model()
            
        # We'll evaluate the SVD model as it provides explicit predictions
        interactions_full_df = self.models['svd']['interactions_full_indexed_df'].reset_index()
        
        # Use k-fold cross-validation
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # Metrics to track
        metrics = {'rmse': [], 'precision': [], 'recall': [], 'f1': []}
        
        for train_index, test_index in kf.split(interactions_full_df):
            # Split data
            train_data = interactions_full_df.iloc[train_index]
            test_data = interactions_full_df.iloc[test_index]
            
            # Create pivot matrix from training data
            train_pivot = train_data.pivot(index='userID', columns='isbn', values='book_rating').fillna(0)
            train_matrix = train_pivot.values
            
            # Perform SVD
            U, sigma, Vt = svds(train_matrix, k=self.config['svd_factors'])
            sigma = np.diag(sigma)
            
            # Make predictions
            predicted_ratings = np.dot(np.dot(U, sigma), Vt)
            preds_df = pd.DataFrame(predicted_ratings, 
                                   index=train_pivot.index, 
                                   columns=train_pivot.columns)
            
            # Create test set for evaluation
            test_pivot = test_data.pivot(index='userID', columns='isbn', values='book_rating').fillna(0)
            
            # Keep only users and items that appear in both train and test sets
            common_users = set(train_pivot.index) & set(test_pivot.index)
            common_items = set(train_pivot.columns) & set(test_pivot.columns)
            
            if not common_users or not common_items:
                print("Warning: No common users or items between train and test sets in this fold. Skipping.")
                continue
                
            test_pivot = test_pivot.loc[list(common_users), list(common_items)]
            preds_df = preds_df.loc[list(common_users), list(common_items)]
            
            # Calculate RMSE
            y_true = test_pivot.values.flatten()
            y_pred = preds_df.values.flatten()
            
            # Remove entries with zero values in the test set
            nonzero_indices = y_true != 0
            y_true = y_true[nonzero_indices]
            y_pred = y_pred[nonzero_indices]
            
            if len(y_true) == 0:
                print("Warning: No non-zero entries in test set. Skipping fold.")
                continue
                
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['rmse'].append(rmse)
            
            # For precision, recall, F1: consider ratings >= 8 as "relevant"
            # and predictions >= 8 as "recommended"
            threshold = 8  # can be adjusted
            y_true_binary = (y_true >= threshold).astype(int)
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calculate precision, recall, F1
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            
        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
        
        print("Model Evaluation Results:")
        for metric, value in avg_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
            
        self.metrics = avg_metrics
        return avg_metrics

# Main execution
if __name__ == "__main__":
    # Initialize the recommender system
    recommender = BookRecommender()
    
    # Build recommendation models
    print("\n===== Building Recommendation Models =====")
    popular_books = recommender.build_popularity_model()
    print("\nTop 10 Popular Books:")
    print(popular_books.head(10))
    
    recommender.build_item_based_cf()
    recommender.build_svd_model()
    recommender.build_content_based_model()
    
    # Example recommendations
    print("\n===== Example Recommendations =====")
    
    # Example user and book
    example_user_id = 69078  # Using an example user ID
    example_book_isbn = '0316666343'  # One of the most rated books
    
    # Get recommendations using different methods
    print("\nPopularity-based recommendations:")
    pop_recs = recommender.get_recommendations_popularity(5)
    print(pop_recs)
    
    print("\nContent-based recommendations for book:", example_book_isbn)
    content_recs = recommender.get_recommendations_content_based(example_book_isbn, 5)
    print(content_recs)
        
    print("\nItem-based CF recommendations for book:", example_book_isbn)
    item_cf_recs = recommender.get_recommendations_item_cf(example_book_isbn, 5)
    print(item_cf_recs)
        
    print("\nSVD-based recommendations for user:", example_user_id)
    svd_recs = recommender.get_recommendations_svd(example_user_id, 5)
    print(svd_recs)
        
    print("\nHybrid recommendations for user:", example_user_id)
    hybrid_recs = recommender.hybrid_recommendations(example_user_id, example_book_isbn, 5)
    print(hybrid_recs)
    
    # Evaluate models
    print("\n===== Model Evaluation =====")
    metrics = recommender.evaluate_models()
    
    print("\nAll processing completed!")