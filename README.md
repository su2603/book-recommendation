
# 📚 Book Recommendation System 

This documentation provides an in-depth explanation of the Book Recommendation System implemented using Python. It combines multiple recommendation strategies, data preprocessing techniques, and utility tools for model inspection and evaluation.

---

## 🔧 Files Overview

### 1. `bookRecommendationSystem.py`
> Main class-based implementation for the recommender system.

### 2. `bookRecSystem.py`
> An exploratory notebook-like script for EDA, preprocessing, and basic modeling.

### 3. `model_extractor.py`
> Utility script for extracting model internals from a saved `.pkl` file.

### 4. `load_model.py`
> Interactive CLI script to load and explore the model from a pickle file.

---

## 📂 `bookRecommendationSystem.py`

### ✅ Features
- Preprocessing for users, books, and ratings data
- Popularity-based recommendation
- Item-based Collaborative Filtering (using KNN)
- Model-based CF (using SVD)
- Content-based recommendation (TF-IDF & cosine similarity)
- Hybrid model (combined output from multiple strategies)
- Model evaluation (RMSE, Precision, Recall, F1)
- Visualizations: ratings, age, country, publication year, authors
- User profile insights
- Book analytics
- Model saving/loading

### 🧼 Data Preprocessing
- Cleans `Age`, `Location`, `Publisher`, `Year-Of-Publication`
- Removes anomalies and fills missing values
- Filters and transforms data for memory efficiency
- Merges datasets into a final dataframe

### 🔍 Recommendation Models
- **Popularity-based**: Weighted average rating for books with enough user interactions
- **Item-based CF**: KNN using cosine similarity on user-book rating matrix
- **SVD-based CF**: Matrix factorization using `scipy.sparse.linalg.svds`
- **Content-based**: TF-IDF on title, author, and publisher metadata
- **Hybrid**: Combines outputs from all above methods

### 📊 Evaluation
- K-Fold CV with metrics: RMSE, Precision, Recall, F1
- Explicit feedback handling only

### 📈 Visualization Tools
- Ratings distribution
- User age histogram
- Publication year histogram
- Top authors and countries

---

## 🧪 `bookRecSystem.py` (EDA + Model Testing Script)

### Purpose
- Conducts detailed data cleaning and visualization
- Builds a simpler popularity and CF model
- Uses item-based KNN and SVD for recommendation
- Evaluates interaction matrix sparsity
- Merges and tests recommender class `CFRecommender`

---

## 🛠️ `model_extractor.py`

### Role
Extracts internal components from a `.pkl` file saved by `BookRecommender`.

### Features
- Parses and prints config and metrics
- Saves model metadata, predictions sample, and components
- Supports:
  - Popularity model parameters and book list
  - SVD predictions matrix shape and stats
  - Content-based vectorizer structure

---

## 🔍 `load_model.py`

### Purpose
Interactive CLI tool to:
- Load a `BookRecommender` model from a pickle file
- Print available components
- Inspect and export nested components to JSON

### Example Usage
```bash
python load_model.py book_recommender.pkl
```

---

## 💾 Saving & Loading Models

### Save
```python
recommender.save_model("book_recommender.pkl")
```

### Load
```python
BookRecommender.load_model("book_recommender.pkl")
```

---

## 📌 Notable Classes & Methods

### `BookRecommender`
- `build_popularity_model()`
- `build_item_based_cf()`
- `build_svd_model()`
- `build_content_based_model()`
- `hybrid_recommendations()`
- `evaluate_models()`
- `visualize_*()`
- `analyze_book(isbn)`
- `get_user_reading_profile(user_id)`

---

## 📤 Export & Interactivity

- `export_recommendations_to_csv(df)`
- CLI tools for inspection (`load_model.py`, `model_extractor.py`)

---

## ✅ Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## 📎 Summary

This modular and well-documented system allows robust experimentation and evaluation of book recommendation strategies using both collaborative and content-based techniques, and is optimized for scalability and clarity.


# 📚 Book Recommendation System Documentation (Extended)

This comprehensive documentation explains the architecture, methodology, components, and usage of a Python-based Book Recommendation System that supports collaborative filtering, content-based filtering, and hybrid models.

---

## 🗂️ File Breakdown

### 1. `bookRecommendationSystem.py` - Core Class
Implements the `BookRecommender` class for a modular, production-grade book recommendation pipeline.

### 2. `bookRecSystem.py` - Exploratory Script
Performs dataset analysis, visualizations, and initial model testing.

### 3. `model_extractor.py` - Model Debug Utility
Loads and extracts internals from serialized recommender models.

### 4. `load_model.py` - CLI Tool
CLI-based inspection and JSON export of a serialized recommendation model.

---

## 🧠 BookRecommender Class

### 🔧 Configuration (default in `self.config`)
| Key                      | Description |
|--------------------------|-------------|
| `popularity_threshold`   | Minimum ratings to consider a book in popularity model |
| `user_interactions_threshold` | Minimum user activity required for SVD |
| `rating_threshold`       | Minimum ratings a user must give to be used in KNN |
| `book_ratings_threshold_perc` | Top % of rated books to keep in KNN |
| `svd_factors`            | Number of latent factors in SVD |
| `content_batch_size`     | Batching size to avoid memory overflow in content model |

---

## 🧹 Preprocessing Stages

### Users
- Normalize age (5–100) and country info
- Extract `Country` from `Location`
- Fill NaN with country-median or global mean

### Books
- Convert `Year-Of-Publication` to int16
- Fix anomalous records by ISBN
- Drop image URLs to save memory

### Ratings
- Remove implicit ratings (rating == 0)
- Add average rating and rating count per book
- Merge all into `self.final_dataset`

---

## 🤖 Recommendation Models

### 1. Popularity-Based Filtering
- Weighted average: `(v/(v+m) * R) + (m/(m+v) * C)`
- Uses 90th percentile rating threshold

### 2. Item-Based Collaborative Filtering (KNN)
- Uses cosine similarity between items
- Ratings matrix: users vs books
- Built with `sklearn.neighbors.NearestNeighbors`

### 3. Matrix Factorization (SVD)
- Filters inactive users
- Applies log smoothing
- Decomposes rating matrix with `scipy.sparse.linalg.svds`

### 4. Content-Based Filtering
- TF-IDF vectorizer on: `Title + Author + Publisher`
- Similarity: Cosine distance with target book vector
- Calculates similarities in batches

### 5. Hybrid Model
- Merges outputs from all other models
- Priority order: Content > Item CF > SVD > Popularity

---

## 📏 Evaluation (`evaluate_models`)
- K-Fold cross-validation (default K=5)
- Metrics:
  - RMSE
  - Precision@K
  - Recall@K
  - F1 Score

---

## 📊 Visualization Tools
| Method                          | Output                         |
|----------------------------------|--------------------------------|
| `visualize_ratings_distribution()` | Barplot of rating frequency |
| `visualize_user_age_distribution()` | Age histogram               |
| `visualize_country_distribution()` | Top 10 countries by users   |
| `visualize_publication_years()`   | Year histogram of books      |
| `visualize_popular_authors()`     | Top 15 authors by count      |

---

## 🔍 Utility Functions

| Function                           | Purpose |
|------------------------------------|---------|
| `analyze_book(isbn)`               | Stats & demographic info on a book |
| `get_user_reading_profile(user_id)`| Personalized stats about a user |
| `find_similar_users(user_id)`      | User-user matching |
| `export_recommendations_to_csv(df)`| Save recommendations as CSV |
| `save_model()` and `load_model()`  | Serialize and restore system state |

---

## 🛠️ `model_extractor.py`

### CLI Usage:
```bash
python model_extractor.py book_recommender.pkl
```

### Extracts:
- Config (`config.json`)
- Metrics (`evaluation_metrics.json`)
- Popularity params and top books
- SVD prediction sample & stats
- Content model component metadata

---

## 🧪 `load_model.py`

### Features:
- Loads model and prints top-level keys
- Supports nested key exploration via `models.popularity.m`
- Can export serializable components to `.json`

### CLI Example:
```bash
python load_model.py book_recommender.pkl
```

---

## 💻 Deployment Notes

- Modularized and optimized for memory constraints
- Can work in notebook or CLI environment
- Easily extensible for:
  - Genre-based filtering
  - Deep learning integration (e.g., Autoencoders)

---

## 📌 Tips for Usage

- Always call `ensure_model("model_type")` before requesting recommendations
- Save model after heavy computation using `save_model()`
- Use `visualize_*()` for data insights
- Analyze new users/books before giving recommendations

---

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

---

## ✅ Summary

This system is designed for robust, explainable, and extensible recommendation workflows. It is ideal for learning, experimentation, and production-grade deployment in recommendation pipelines.

