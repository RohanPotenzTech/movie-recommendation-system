import gradio as gr

def recommend_movies(movie_name):
    df = get_recommendation(movie_name)
    if isinstance(df, pd.DataFrame):
        return df.to_string(index=False)
    else:
        return df
app = gr.Interface(
    fn = recommend_movies,
    inputs = "text",
    outputs = "text",
    title = "Movie Recommendation System ",
    description = "Enter the movie name "
)
app.launch()        


import pandas as pd
import numpy as np

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

movies.head()

ratings.head()

final_dataset = ratings.pivot(index = "movieId", columns = "userId", values = "rating")

final_dataset.head()

final_dataset.fillna(0, inplace =True)

final_dataset.head()

no_user_voted = ratings.groupby("movieId")["rating"].agg('count')
no_movies_voted = ratings.groupby("userId")["rating"].agg('count')

import matplotlib.pyplot as plt
plt.style.use("ggplot")
fig,axes = plt.subplots(1,1, figsize=(16,4))
plt.scatter(no_user_voted.index, no_user_voted, color= "blue")
plt.axhline(y=10, color = 'green')
plt.xlabel("MovieId")
plt.ylabel("No of user voted")
plt.show()


final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]

final_dataset


import matplotlib.pyplot as plt
plt.style.use("ggplot")
fig,axes = plt.subplots(1,1, figsize=(16,4))
plt.scatter(no_movies_voted.index, no_movies_voted, color= "blue")
plt.axhline(y=50, color = 'green')
plt.xlabel("User Id")
plt.ylabel("No of user voted")
plt.show()

final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

final_dataset.shape

final_dataset.head()

sample = np.array([[1,0,0,0,0], [0,0,2,0,0], [0,0,4,0,0]])
sparsity = 1.0 - (np.count_nonzero(sample) / float(sample.size))
print(sparsity)

from scipy.sparse import csr_matrix
csr_sample = csr_matrix(sample)
print(csr_sample)


csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace= True)
print(csr_data)


from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 20, n_jobs = -1)
knn.fit(csr_data)

movies.head()

def get_recommendation(movie_name):
    movie_list = movies[movies['title'].str.contains(movie_name)]
    
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        
        distance, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors = 10+1)
        rec_movies_indices = sorted(list(zip(indices.squeeze().tolist(), distance.squeeze().tolist())), key=lambda x: x[1])[:0: -1]
        
        recommended_movies = []
        for val in rec_movies_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommended_movies.append({'Title': movies.iloc[idx] ['title'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(recommended_movies, index = range(1,11))
        return df   
                               
    else:
        return "Movies Not Found"

get_recommendation("Iron Man")




#orginal

def get_recommendation(movie_name):
    movie_list = movies[movies['title'].str.contains(movie_name) | movies['genres'].str.contains(genres)]
    
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        
        distance, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors = 10+1)
        rec_movies_indices = sorted(list(zip(indices.squeeze().tolist(), distance.squeeze().tolist())), key=lambda x: x[1])[:0: -1]
        
        recommended_movies = []
        for val in rec_movies_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommended_movies.append({'Title': movies.iloc[idx] ['title'].values[0], 'Distance': val[1], 'Generes'})
        df = pd.DataFrame(recommended_movies, index = range(1,11))
        return df   
                               
    else:
        return "Movies Not Found"
    


    def get_movie_recommendations(input_value):
    if isinstance(input_value, str):
        movie_list = movies[movies['title'].str.contains(input_value, case=False)]
    else:
        movie_list = movies[movies['genres'].str.contains(input_value, case=False)]
    
    if len(movie_list):
        return movie_list[['title', 'movieId']].head(10)
    else:
        return "No movies found."
    
    print(get_movie_recommendations("Animation"))