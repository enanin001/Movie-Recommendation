# Movie Recommendation System Code Structure

```
.
├── data/                   # Raw data directory
│   ├── movies.csv         # Movie metadata 
│   ├── ratings.csv        # User rating records 
│   └── tags.csv           # User-generated movie tags
│
├── notebooks/             # Analysis notebooks
│   └── test.ipynb         # Exploratory Data Analysis notebook
│
└── README.md              # Project documentation
```

## Core Module Descriptions

### 1. Data Processing 
- Key Features:
  - Handle missing values (fill genres with 'Unknown')
  - Multi-label genre encoding (MultiLabelBinarizer)
  - Temporal feature extraction (year/month/day_of_week)
  - TF-IDF tag feature generation

### 2. Collaborative Filtering 
- Implemented Methods:
  ```python
  # User similarity matrix calculation
  user_sim_matrix = cosine_similarity(user_movie_matrix)
  
  # Item similarity matrix calculation
  item_sim_matrix = cosine_similarity(user_movie_matrix.T)
  ```

### 3. Matrix Factorization 
- Supported Algorithms:
  - **SVD**: Implemented via Surprise library
  - **NMF**: Non-negative Matrix Factorization
  ```python
  # Usage example
  from surprise import SVD
  model = SVD(n_factors=50, n_epochs=20)
  ```

### 4. Neural Network Model 
- Model Architecture:
  ```python
  # Neural network layers
  user_embedding = Embedding(n_users, 50)(user_input)
  item_embedding = Embedding(n_items, 50)(item_input)
  concat = Concatenate()([Flatten()(user_embedding), Flatten()(item_embedding)])
  x = Dense(64, activation='relu')(concat)
  ```

