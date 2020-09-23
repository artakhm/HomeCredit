import argparse
import pickle
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.pickle', help='model for prediction')
    parser.add_argument('--data', type=str, default='data.pickle', help='data to predict')
    parser.add_argument('--out', type=str, default='prediction.csv', help='where to save prediction')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    data = pd.read_csv(args.data)
    if data.isna().sum().sum():
        raise ValueError("Data for prediction has NaN fields")

    y_pred = model.predict_proba(data)[:, 1]
    y_pred = pd.Series(y_pred)
    y_pred.to_csv(args.out)

    print('Prediction has successfully saved in \'' + args.out + '\'')


if __name__ == '__main__':
    main()
