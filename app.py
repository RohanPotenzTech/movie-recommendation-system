import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

#print(movies.head())
#print(ratings.head())

movies['genres'] = movies['genres'].str.split('|')
print(movies)

average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index(name='avg_rating')
rating_counts = ratings.groupby('movieId')['rating'].count().reset_index(name='rating_count')

ratings_summary = pd.merge(average_ratings, rating_counts, on='movieId')
print(ratings_summary)

filtered_ratings = ratings_summary[ratings_summary['rating_count'] >= 2]

print(filtered_ratings)

movies_with_ratings = pd.merge(movies, ratings_summary, on = 'movieId', how = 'left')

movies_with_ratings['avg_rating'].fillna(0,inplace=True)
movies_with_ratings['rating_count'].fillna(0,inplace=True)

print(movies_with_ratings)

final_filtered_movies = movies_with_ratings[
    (movies_with_ratings['avg_rating'] >= 2) &
    (movies_with_ratings['rating_count'] >= 5)
].sort_values(by='avg_rating', ascending=False)

print(final_filtered_movies)

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Step 2: Calculate the similarity matrix (cosine similarity between movies)
movie_similarity_matrix = cosine_similarity(user_movie_matrix.T)

# Step 3: Function to get similar movies based on collaborative filtering
def get_collaborative_similar_movies(movie_id, num_recommendations=5):
    movie_idx = user_movie_matrix.columns.get_loc(movie_id)
    similarity_scores = movie_similarity_matrix[movie_idx]
    
    similar_movies = [(user_movie_matrix.columns[i], similarity_scores[i]) for i in range(len(similarity_scores))]
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    
    return similar_movies[1:num_recommendations+1]

# Step 4: Function to search movies by name or genre and provide recommendations
# Step 4: Function to search movies by name or genre and provide recommendations
def search_movies():
    print("Choose an option:")
    print("1: Search By Movie Name")
    print("2: Search By Genre")

    try:
        choice = int(input("Enter your choice (1 or 2): "))
        if choice == 1:
            title_keyword = input("Enter a movie name or keyword: ").strip()
            filtered_movies = movies[movies['title'].str.contains(title_keyword, case=False, na=False)]
        elif choice == 2:
            genre_keyword = input("Enter a genre (e.g., Comedy, Action): ").strip()
            # Handle genres stored as lists
            filtered_movies = final_filtered_movies[final_filtered_movies['genres'].apply(
                lambda genres: genre_keyword.lower() in [g.lower() for g in genres] if isinstance(genres, list) else False
            )]
        else:
            print("Invalid choice. Please enter 1 or 2.")
            return 
        
        if not filtered_movies.empty:
            print("\nFiltered Movies:")
            print(filtered_movies[['movieId', 'title', 'genres']])

            # For each filtered movie, get collaborative recommendations
            print("\nCollaborative Recommendations based on the filtered movie(s):")
            count = 0  # Counter for limiting the number of recommendations
            for movie_id in filtered_movies['movieId']:
                if count >= 10:  # Stop after showing 10 recommendations
                    break
                recommended_movies = get_collaborative_similar_movies(movie_id)
                for recommended_movie, score in recommended_movies:
                    if count >= 10:
                        break
                    recommended_movie_data = movies[movies['movieId'] == recommended_movie]
                    print(f"Movie: {recommended_movie_data['title'].values[0]}, Similarity Score: {score}")
                    count += 1
        else:
            print("\nNo movies found matching your criteria.")
    except ValueError:
        print("Please enter a valid number (1 or 2).")

# Step 5: Run the search_movies function
search_movies()
