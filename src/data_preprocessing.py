import pandas as pd  
from sklearn.preprocessing import MultiLabelBinarizer  

def load_data(data_path="data/"):  
    ratings = pd.read_csv(f"{data_path}/ratings.csv")  
    movies = pd.read_csv(f"{data_path}/movies.csv")  
    return ratings, movies  

def preprocess_movies(movies):  
    # 提取电影发布年份  
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)  
    # 多热编码电影类型  
    mlb = MultiLabelBinarizer()  
    genres_encoded = mlb.fit_transform(movies['genres'].str.split('|'))  
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)  
    return pd.concat([movies, genres_df], axis=1)  