import pandas as pd
from surprise import Dataset, Reader
from surprise import accuracy

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

def evaluate(model, test_data):
    test_set = test_data.build_full_trainset().build_testset()
    predictions = model.test(test_set)
    metric = accuracy.rmse(predictions)
    return metric

def main():
    # File paths for the trained model and testing dataset
    model_file = '/kaggle/working/PMLDL_movie_recommender_system/models/model.pkl'
    test_file = '/kaggle/working/PMLDL_movie_recommender_system/data/interim/preprocessed_data.csv'

    test_data = load_data(test_file)

    trained_model = load_surprise_model(model_file)['algo']

    surprise_test_data = load_surprise_data(test_data)

    # Evaluate the model
    benchmark_score = evaluate(trained_model, surprise_test_data)

    print(f'Benchmark Score: {benchmark_score}')

if __name__ == "__main__":
    main()
