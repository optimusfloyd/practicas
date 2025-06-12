import pandas as pd
from models.naive_model import NaiveModel

INPUT_DATA_PATH = 'mnist_784.csv'
MODEL_PATH = 'naive_model.pkl'
OUTPUT_DATA_PATH = 'mnist_784_predictions.csv'

def main():
    df = pd.read_csv(INPUT_DATA_PATH)
    model = NaiveModel()
    model.load(MODEL_PATH)
    predictions = model.predict(df)
    predictions.to_csv(OUTPUT_DATA_PATH, index=False)

if __name__ == '__main__':
    main()

