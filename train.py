import pandas as pd
import numpy as np
import argparse
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

def main(args):
    if args.mode == "train":
        train_data = pd.read_csv(args.train_path)

        X = train_data.drop('model_output', axis=1)
        # print(X)
        y = train_data['model_output'].astype(int)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=True)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True)

        model.save_model('model.pth')
        print("Model training completed")
    else:
        model = CatBoostClassifier()
        model.load_model('model.pth')

        test_data = pd.read_csv(args.test_path)
        predictions = model.predict_proba(test_data)[:, 1]

        num_per_row = 15
        num_rows = len(predictions) // num_per_row
        reshaped_predictions = predictions[:num_rows * num_per_row].reshape((num_rows, num_per_row))

        columns = ['p' + str(i+1) for i in range(num_per_row)]
        results = pd.DataFrame(reshaped_predictions, columns=columns)
        results.to_csv(args.outputs, index_label='id')
        print(f"Predictions saved to {args.outputs}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--outputs', type=str)
    args = parser.parse_args()
    main(args)
