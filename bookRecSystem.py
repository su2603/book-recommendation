# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')

# Load datasets
users = pd.read_csv('Users.csv')
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')

# Check dimensions of datasets
print(f'''\t  Book_df shape is {books.shape}
          Ratings_df shape is {ratings.shape}
          Users_df shape is {users.shape}''')

# Define function to check missing values
def missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_percent = round(df.isnull().mean().mul(100), 2)
    mz_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns={df.index.name:'col_name', 0:'Missing Values', 1:'% of Total Values'})
    mz_table['Data_type'] = df.dtypes
    mz_table = mz_table.sort_values('% of Total Values', ascending=False)
    return mz_table.reset_index()

# Analyze Users Dataset
missing_values(users)

# Age Distribution
plt.figure(figsize=(10, 6))
users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100], color='skyblue', edgecolor='black')
plt.title('Age Distribution', fontsize=15)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Check for outliers in Age column
plt.figure(figsize=(8, 6))
sns.boxplot(y='Age', data=users, palette='viridis')
plt.title('Find outlier data in Age column', fontsize=15)
plt.show()

print(sorted(users.Age.unique()))

# Create Country column from Location
for i in users:
    users['Country'] = users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')   

users.drop('Location', axis=1, inplace=True)
users['Country'] = users['Country'].astype('str')

# Clean Country data
users['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                         ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)

# Plot country distribution
plt.figure(figsize=(15, 7))
sns.countplot(y='Country', data=users, order=pd.value_counts(users['Country']).iloc[:10].index, palette='viridis')
plt.title('Count of users Country wise', fontsize=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.show()

# Handle age outliers
plt.figure(figsize=(10, 6))
sns.distplot(users.Age, color='purple', kde=True)
plt.title('Age Distribution Plot', fontsize=15)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(linestyle='--', alpha=0.7)
plt.show()

# Replace extreme ages with NaN
users.loc[(users.Age > 100) | (users.Age < 5), 'Age'] = np.nan

# Fill NaN values with median by country
users['Age'] = users['Age'].fillna(users.groupby('Country')['Age'].transform('median'))

# Fill remaining NaN with mean
users['Age'].fillna(users.Age.mean(), inplace=True)

# Books Dataset Analysis
# Top 10 authors with most books
plt.figure(figsize=(15, 7))
sns.countplot(y='Book-Author', data=books, order=pd.value_counts(books['Book-Author']).iloc[:10].index, palette='magma')
plt.title('Top 10 Authors', fontsize=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Author', fontsize=12)
plt.show()

# Top 10 publishers
plt.figure(figsize=(15, 7))
sns.countplot(y='Publisher', data=books, order=pd.value_counts(books['Publisher']).iloc[:10].index, palette='plasma')
plt.title('Top 10 Publishers', fontsize=15)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Publisher', fontsize=12)
plt.show()

# Clean Year-Of-Publication data
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype('str')
books.loc[books.ISBN == '0789466953', 'Year-Of-Publication'] = 2000
books.loc[books.ISBN == '0789466953', 'Book-Author'] = "James Buckley"
books.loc[books.ISBN == '0789466953', 'Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953', 'Book-Title'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

books.loc[books.ISBN == '078946697X', 'Year-Of-Publication'] = 2000
books.loc[books.ISBN == '078946697X', 'Book-Author'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X', 'Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X', 'Book-Title'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

books.loc[books.ISBN == '2070426769', 'Year-Of-Publication'] = 2003
books.loc[books.ISBN == '2070426769', 'Book-Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769', 'Publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769', 'Book-Title'] = "Peuple du ciel, suivi de 'Les Bergers"

# Convert Year-Of-Publication to numeric
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')

# Handle invalid years
books.loc[(books['Year-Of-Publication'] > 2006) | (books['Year-Of-Publication'] == 0), 'Year-Of-Publication'] = np.nan
books['Year-Of-Publication'].fillna(round(books['Year-Of-Publication'].median()), inplace=True)

# Drop image URL columns
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)

# Fill NaN values in Publisher and Book-Author
books.Publisher.fillna('other', inplace=True)
books['Book-Author'].fillna('other', inplace=True)

# Ratings Dataset Analysis
# Keep only ratings for books that exist in books dataset
ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
print("Shape of dataset before filtering:", ratings.shape)
print("Shape of dataset after filtering:", ratings_new.shape)

# Keep only ratings from users who exist in users dataset
ratings_new = ratings_new[ratings_new['User-ID'].isin(users['User-ID'])]
print("Shape of dataset after ensuring users exist:", ratings_new.shape)

# Plot rating distribution
plt.figure(figsize=(12, 6))
ratings_new['Book-Rating'].value_counts(sort=False).plot(kind='bar', color=sns.color_palette("viridis", 10))
plt.title('Rating Distribution', fontsize=15)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Separate implicit and explicit ratings
ratings_explicit = ratings_new[ratings_new['Book-Rating'] != 0]
ratings_implicit = ratings_new[ratings_new['Book-Rating'] == 0]
print('Explicit ratings dataset shape:', ratings_explicit.shape)
print('Implicit ratings dataset shape:', ratings_implicit.shape)

# Plot explicit ratings distribution
plt.figure(figsize=(12, 8))
sns.countplot(data=ratings_explicit, x='Book-Rating', palette='rocket_r')
plt.title('Explicit Ratings Distribution', fontsize=15)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Find top 5 books by rating count
rating_count = pd.DataFrame(ratings_explicit.groupby('ISBN')['Book-Rating'].count())
print("Top 5 books by rating count:")
print(rating_count.sort_values('Book-Rating', ascending=False).head())

# Get details of the most rated books
most_rated_books = pd.DataFrame(['0316666343', '0971880107', '0385504209', '0312195516', '0060928336'], 
                               index=np.arange(5), columns=['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
print("Details of the most rated books:")
print(most_rated_books_summary)

# Create aggregate columns for ratings
ratings_explicit['Avg_Rating'] = ratings_explicit.groupby('ISBN')['Book-Rating'].transform('mean')
ratings_explicit['Total_No_Of_Users_Rated'] = ratings_explicit.groupby('ISBN')['Book-Rating'].transform('count')

# Merge all datasets
Final_Dataset = users.copy()
Final_Dataset = pd.merge(Final_Dataset, ratings_explicit, on='User-ID')
Final_Dataset = pd.merge(Final_Dataset, books, on='ISBN')

# Popularity Based Filtering
C = Final_Dataset['Avg_Rating'].mean()
m = Final_Dataset['Total_No_Of_Users_Rated'].quantile(0.90)
Top_Books = Final_Dataset.loc[Final_Dataset['Total_No_Of_Users_Rated'] >= m]
print(f'C={C} , m={m}')
print("Top books dataset shape:", Top_Books.shape)

def weighted_rating(x, m=m, C=C):
    v = x['Total_No_Of_Users_Rated']
    R = x['Avg_Rating']
    return (v/(v+m) * R) + (m/(m+v) * C)
    
Top_Books['Score'] = Top_Books.apply(weighted_rating, axis=1)
Top_Books = Top_Books.sort_values('Score', ascending=False)

# Keep unique ISBNs only
Top_Books = Top_Books.sort_values('Score', ascending=False).drop_duplicates('ISBN').sort_index()
Top_Books = Top_Books.sort_values('Score', ascending=False)

# Display top 20 books
print("Top 20 recommended books based on popularity:")
print(Top_Books.reset_index(drop=True).head(20))

# Model-Based Collaborative Filtering
# Filter users with at least 3 ratings
ratings_explicit.rename(columns={'User-ID':'user_id', 'ISBN':'isbn', 'Book-Rating':'book_rating'}, inplace=True)
user_ratings_threshold = 3

filter_users = ratings_explicit['user_id'].value_counts()
filter_users_list = filter_users[filter_users >= user_ratings_threshold].index.to_list()

df_ratings_top = ratings_explicit[ratings_explicit['user_id'].isin(filter_users_list)]
print('Filter: users with at least %d ratings\nNumber of records: %d' % (user_ratings_threshold, len(df_ratings_top)))

# Filter top 10% most rated books
book_ratings_threshold_perc = 0.1
book_ratings_threshold = len(df_ratings_top['isbn'].unique()) * book_ratings_threshold_perc

filter_books_list = df_ratings_top['isbn'].value_counts().head(int(book_ratings_threshold)).index.to_list()
df_ratings_top = df_ratings_top[df_ratings_top['isbn'].isin(filter_books_list)]
print('Filter: top %d%% most frequently rated books\nNumber of records: %d' % (book_ratings_threshold_perc*100, len(df_ratings_top)))

# Item-based Collaborative Filtering using KNN
df_ratings_top.rename(columns={'user_id':'userID', 'isbn':'ISBN', 'book_rating':'bookRating'}, inplace=True)

# Generate ratings matrix
ratings_matrix = df_ratings_top.pivot(index='userID', columns='ISBN', values='bookRating')
userID = ratings_matrix.index
ISBN = ratings_matrix.columns
print("Ratings matrix shape:", ratings_matrix.shape)

n_users = ratings_matrix.shape[0]
n_books = ratings_matrix.shape[1]
print("Number of users:", n_users, "Number of books:", n_books)

# Fill NaN values with zeros
ratings_matrix.fillna(0, inplace=True)
ratings_matrix = ratings_matrix.astype(np.int32)

# Calculate sparsity
sparsity = 1.0 - len(ratings_explicit)/float(ratings_explicit.shape[0]*n_books)
print(f'The sparsity level of Book Crossing dataset is {sparsity*100} %')

# Create combined dataset for popularity filtering
combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['Book-Author', 'Year-Of-Publication', 'Publisher']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
combine_book_rating.rename(columns={'User-ID':'userID', 'Book-Title':'bookTitle', 'Book-Rating':'bookRating'}, inplace=True)

combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])

# Count ratings per book
book_ratingcount = (combine_book_rating.
                   groupby(by=['bookTitle'])['bookRating'].
                   count().
                   reset_index().
                   rename(columns={'bookRating':'TotalRatingCount'})
                   )

# Merge rating data with count data
rating_with_totalratingcount = combine_book_rating.merge(book_ratingcount, left_on='bookTitle', right_on='bookTitle', how='inner')

# Filter popular books (≥ 50 ratings)
popularity_threshold = 50
rating_popular_book = rating_with_totalratingcount.query('TotalRatingCount >= @popularity_threshold')

# Remove duplicates if any
if not rating_popular_book[rating_popular_book.duplicated(['userID', 'bookTitle'])].empty:
    initial_rows = rating_popular_book.shape[0]
    rating_popular_book = rating_popular_book.drop_duplicates(['userID', 'bookTitle'])
    current_rows = rating_popular_book.shape[0]
    print(f'Removed {initial_rows - current_rows} duplicate rows')

# Create pivot table for KNN
from scipy.sparse import csr_matrix  # Import csr_matrix

us_canada_user_rating_pivot = rating_popular_book.pivot(index='bookTitle', columns='userID', values='bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

# Build KNN model
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(us_canada_user_rating_matrix)

# SVD-based Collaborative Filtering
ratings_explicit.rename(columns={'user_id':'User-ID', 'isbn':'ISBN', 'book_rating':'Book-Rating'}, inplace=True)

# Filter users with many interactions
users_interactions_count_df = ratings_explicit.groupby(['ISBN', 'User-ID']).size().groupby('User-ID').size()
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 100].reset_index()

interactions_from_selected_users_df = ratings_explicit.merge(users_with_enough_interactions_df, 
                                                           how='right',
                                                           left_on='User-ID',
                                                           right_on='User-ID')

# Apply log transformation to user preferences
import math
def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df.groupby(['ISBN', 'User-ID'])['Book-Rating'].sum().apply(smooth_user_preference).reset_index()

# Split data into train and test
from sklearn.model_selection import train_test_split
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                              stratify=interactions_full_df['User-ID'], 
                                                              test_size=0.20,
                                                              random_state=42)

# Create pivot table
users_items_pivot_matrix_df = interactions_train_df.pivot(index='User-ID', 
                                                          columns='ISBN', 
                                                          values='Book-Rating').fillna(0)

users_items_pivot_matrix = users_items_pivot_matrix_df.values
users_ids = list(users_items_pivot_matrix_df.index)

# SVD decomposition
from scipy.sparse.linalg import svds
NUMBER_OF_FACTORS_MF = 15
U, sigma, Vt = svds(users_items_pivot_matrix, k=NUMBER_OF_FACTORS_MF)
sigma = np.diag(sigma)

# Reconstruct the prediction matrix
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns=users_items_pivot_matrix_df.columns, index=users_ids).transpose()

# Recommender class
class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df):
        self.cf_predictions_df = cf_predictions_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['ISBN'].isin(items_to_ignore)].sort_values('recStrength', ascending=False).head(topn)
        recommendations_df = recommendations_df.merge(books, on='ISBN', how='inner')
        recommendations_df = recommendations_df
        return recommendations_df

cf_recommender_model = CFRecommender(cf_preds_df)

# Index dataframes for faster lookup
interactions_full_indexed_df = interactions_full_df.set_index('User-ID')
interactions_train_indexed_df = interactions_train_df.set_index('User-ID')
interactions_test_indexed_df = interactions_test_df.set_index('User-ID')

# Function to get items a user has interacted with
def get_items_interacted(UserID, interactions_df):
    interacted_items = interactions_df.loc[UserID]['ISBN']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

# Recommender class for model evaluation
class ModelRecommender:
    def get_not_interacted_items_sample(self, UserID, sample_size, seed=42):
        interacted_items = get_items_interacted(UserID, interactions_full_indexed_df)
        all_items = set(ratings_explicit['ISBN'])
        non_interacted_items = all_items - interacted_items
        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index
    
    def evaluate_model_for_user(self, model, person_id):
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        
        if type(interacted_values_testset['ISBN']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
            person_interacted_items_testset = set([interacted_values_testset['ISBN']])
            
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id, interactions_train_indexed_df), topn=10000000000)
        print('Recommendation for User-ID =', person_id)
        print(person_recs_df.head(10))

    def recommend_book(self, model, userid):
        self.evaluate_model_for_user(model, userid)

model_recommender = ModelRecommender()

# Get a user ID from the available users
print("Available user IDs for recommendation:")
print(list(interactions_full_indexed_df.index.values)[:10], "...")  # Just show first 10 IDs

# Function to get recommendations for a specific user
def get_recommendations_for_user(user_id):
    model_recommender.recommend_book(cf_recommender_model, user_id)
    
# Example: Get recommendations for user 69078
get_recommendations_for_user(69078)