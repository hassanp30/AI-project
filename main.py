import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


FILE_PATH = Path(__file__).resolve().parent



def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

        

class NumPyRNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.Wxh = np.random.randn(input_dim, hidden_dim) 
        self.Whh = np.random.randn(hidden_dim, hidden_dim)
        self.Wyh = np.random.randn(hidden_dim, output_dim)

        self.bh = np.zeros(hidden_dim)
        self.by = np.zeros(output_dim)



    def forward(self, X):

        batch_size = X.shape[0]
        window_length = X.shape[1]

        hs = {}
        hs[-1] = np.zeros(shape=(self.hidden_dim, batch_size))

        for i in range(window_length):
            h = np.tanh(self.Wxh @ X[:,i] + self.Whh @ hs[i-1] + self.bh[:, np.newaxis])
            hs[i] = h
        
        logits = hs[window_length - 1] @ self.Wyh + self.by
        probs = softmax(logits)
        return probs, (X, hs, probs)
            
    def backward(self, Y_true, probs, hs):

    
    def update_params(self, gradients): 



def create_windows(data, window_size=200, step_size = 100):
    """
    creates windows for the dataset.
    returns X and Y as np arrays.
    """
    X_all = []
    Y_all = []
    for _, group in data.groupby('user'):
        X_labels = []
        Y_labels = []

        features = group[['x','y','z']].values
        activities = group['activity'].values
        for i in range(0, len(group) - window_size+1, step_size):
            dominant_label = Counter(activities[i:i+window_size]).most_common(1)[0][0]
            X_labels.append(features[i:i+window_size])
            Y_labels.append(dominant_label)
        
        if len(X_labels) > 0:
            X_all.append(np.array(X_labels))
            Y_all.append(np.array(Y_labels))

    X = np.concatenate(X_all, axis = 0)
    Y = np.concatenate(Y_all, axis = 0)
    return X, Y


def load_dataset():

    """
    returns dataset, train_dataset, test_dataset as pandas DataFrame
    """
    dataset_path = FILE_PATH / "dataset" / "WISDM_ar_v1.1_cleaned.csv"
    dataset = pd.read_csv(dataset_path,
        dtype={
            'user': int,
            'timestamp': float,
            'x':float,
            'y':float,
            'z':float
        }
    )
    # print(type(dataset))

    train_dataset = dataset[(dataset['user']) <= 25]
    test_dataset = dataset[dataset['user'] > 25]

    train_features = train_dataset[['x', 'y', 'z']].values
    test_features = test_dataset[['x', 'y', 'z']].values
    
    # print(train_features[0:10])

    # standardizing the data
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)

    train_dataset[['x', 'y', 'z']] = (train_features - mean) / std

    test_dataset[['x', 'y', 'z']] = (test_features - mean) / std

    return dataset, train_dataset, test_dataset


def main():
    # loading the dataset (fixed the data set using another script).
    dataset, train_dataset, test_dataset = load_dataset()
    
    # windowing the dataset 
    X_train, y_train = create_windows(train_dataset)
    X_test, y_test = create_windows(test_dataset)
    


    

if __name__ == "__main__":
    main()