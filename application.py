from flask import render_template, request, Flask
import pandas as pd
import numpy as np
import os

def create_app():
    application = Flask(__name__,
                static_url_path='',
                static_folder='static',
                template_folder='templates')
    application.config['SECRET_KEY'] = 'AFLDKJFDLSFJSLFDNDCEBE'
    return application

application = create_app()

def calculate_cosine_sim(series1, series2):
    s1 = series1.dropna()
    s2 = series2.dropna()
    intersection = pd.concat([s1, s2], join="inner", axis=1)
    arr1 = intersection.iloc[:, 0].to_numpy()
    arr2 = intersection.iloc[:, 1].to_numpy()
    top = np.sum(arr1 * arr2)
    bot = np.sqrt(np.sum(arr1 ** 2)) * np.sqrt(np.sum(arr2 ** 2))
    if bot == 0:
        return 0
    sim = top/bot
    sim = (1 + sim)/2
    return sim

# Takes as input the UBCF matrix with top k neighbors, and then output a list of predicted ratings
def pred_ratings(pred_matrix):
    sim_index = len(pred_matrix.columns) - 1
    pred_ratings = []
    for i in range(0, sim_index):
        cur_col = pred_matrix.iloc[:, [i, sim_index]].dropna()
        numerator = 0
        denominator = 0
        for index, row in cur_col.iterrows():
            # iloc[0] is the rating, iloc[1] is the Simularity
            numerator += row.iloc[0] * row.iloc[1]
            denominator += row.iloc[1]

        weighted_sum = float("NaN")
        if denominator != 0:
            weighted_sum = numerator/denominator
        pred_ratings.append(weighted_sum)
    return pred_ratings

@application.route('/')
def home():
    # Recommend based on Genre
    genre = request.args.get('genre')
    baseDir = os.getcwd() + "/static/"
    if genre is not None:
        rec = pd.read_csv(baseDir + genre + ".csv")
        movies = rec["Title"].to_numpy()
        ratings = rec["AverageRating"].to_numpy()
        movie_genres = rec["Genres"].to_numpy()
        return render_template("index.html", genre=genre, movies=movies, ratings=ratings, movie_genres=movie_genres,
                               movies_len=len(movies))
    return render_template("index.html", genre=genre)


@application.route('/rec-rating', methods=['GET', 'POST'])
def rec_rating():
    # Recommend based on Rating
    baseDir = os.getcwd() + "/static/"
    movies_csv = pd.read_csv(baseDir + "movies.csv")
    movie_indices = movies_csv["MovieID"].to_numpy()
    movie_names = movies_csv["Title"].to_numpy()
    if request.method == 'POST':
        data = request.form
        if data is not None:
            ids = []
            ratings = []
            for (key, value) in data.items():
                ids.append(int(key))
                ratings.append(int(value))
            if len(ids) == 0:
                return render_template("rec-rating-results.html", movies_len=0)
            mean_rating = np.mean(ratings)
            ratings = ratings - mean_rating
            user_rating_mapping = pd.read_csv(baseDir + "user_movie_mapping.csv")
            user_rating_mapping.columns = user_rating_mapping.columns.astype(int)
            test_data = pd.Series(data=ratings, index=ids)
            #print(test_data)
            # Calculate similarity to test_data
            sims = []
            for i in range(0, user_rating_mapping.shape[0]):
                sim = calculate_cosine_sim(user_rating_mapping.iloc[i], test_data)
                sims.append(sim)

            user_rating_mapping["Sim"] = sims

            # Sort by simularity
            user_rating_mapping = user_rating_mapping.sort_values(by=['Sim'], ascending=False)
            # Get top 20 similar users
            top_20 = user_rating_mapping.head(20)
            # print(top_20)

            # Get indices of movies the user did not rate
            test_na_indices = movie_indices[~np.in1d(movie_indices, test_data.index.to_numpy())]
            # For the not rated entries, need to calculate weighted averaged from the top neighbors
            pred_matrix = top_20[test_na_indices]
            pred_matrix["Sim"] = top_20["Sim"]

            # predict missing values
            pred = np.array(pred_ratings(pred_matrix))
            # adding back the test user mean to the pred ratings
            pred = pred + mean_rating
            ubcf_results = pd.Series(data=pred, index=test_na_indices)
            non_na_predictions = ubcf_results.dropna()
            non_na_predictions = non_na_predictions.sort_values(ascending=False)
            #print(non_na_predictions)

            predicted_movies_ids = non_na_predictions.index.to_numpy()
            ids_to_titles = pd.Series(data=movie_names, index=movie_indices)
            predicted_movie_titles = ids_to_titles[predicted_movies_ids].to_numpy()
            # predicted_ratings = non_na_predictions.to_numpy()
            return render_template("rec-rating-results.html", movies_len=len(predicted_movie_titles),
                                   movies=predicted_movie_titles)
    return render_template("rec-rating.html", movies_len=len(movie_indices), movie_indices=movie_indices,
                           movie_names=movie_names)

if __name__ == '__main__':
    application.run(debug=False)