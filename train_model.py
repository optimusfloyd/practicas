import pandas as pd
from models.naive_model import NaiveModel

DATA_PATH = 'mnist_784.csv'
MODEL_PATH = 'naive_model.pkl'

def main():
    df = pd.read_csv(DATA_PATH)
    model = NaiveModel()
    model.fit(df)
    model.save(MODEL_PATH)

if __name__ == '__main__':
    main()

