import pandas as pd
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate
import joblib

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def load_surprise_data(data):
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    return surprise_data

def load_surprise_model(model_path):
    model = joblib.load(model_path)
    return model

def evaluate(model, data):
    results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return results

def main():
    # File paths for the trained model and testing dataset
    model_file = '/kaggle/working/PMLDL_movie_recommender_system/models/model.pkl'
    data_file = '/kaggle/working/PMLDL_movie_recommender_system/data/interim/preprocessed_data.csv'

    data = load_data(data_file)

    trained_model = load_surprise_model(model_file)['algo']

    surprise_test_data = load_surprise_data(data)

    # Evaluate the model
    benchmark_score = evaluate(trained_model, surprise_test_data)

if __name__ == "__main__":
    main()
