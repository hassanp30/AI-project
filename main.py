from numpy import gradient
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

np.random.seed(42)
try:
    FILE_PATH = Path(__file__).resolve().parent
except NameError:
    FILE_PATH = Path.cwd()

ACTIVITY_MAP = {
    'Walking': 0,
    'Jogging': 1,
    'Upstairs': 2,
    'Downstairs': 3,
    'Sitting': 4,
    'Standing': 5
    }
INT_TO_ACTIVITY = {v: k for k, v in ACTIVITY_MAP.items()}



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

        scale = 0.01
        self.Wxh = np.random.randn(input_dim, hidden_dim) * scale
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.Wyh = np.random.randn(hidden_dim, output_dim) * scale

        self.bh = np.zeros(hidden_dim)
        self.by = np.zeros(output_dim)



    def forward(self, X):

        batch_size = X.shape[0]
        window_length = X.shape[1]

        hs = {}
        hs[-1] = np.zeros(shape=(batch_size, self.hidden_dim))

        for i in range(window_length):
            h = np.tanh(X[:,i] @ self.Wxh + hs[i-1] @ self.Whh + self.bh)   # dim: (batch_size, hidden_dim)
            hs[i] = h
        
        logits = hs[window_length - 1] @ self.Wyh + self.by
        probs = softmax(logits)
        return probs, (X, hs, probs)

    @staticmethod
    def cross_entropy_loss(probs, y_true):
        epsilon = 1e-12        
        n = probs.shape[0]
        correct_probs = probs[np.arange(n), y_true]
        loss = -np.mean(np.log(correct_probs + epsilon))
        return loss


    def backward(self, y_true, cache):
        X, hs, probs = cache
        
        num_sample = X.shape[0]
        window_length = X.shape[1]
        
        dlogits = probs.copy()
        dlogits[np.arange(num_sample), y_true] -= 1
        dlogits /= num_sample

        h_final = hs[window_length - 1]
        dWyh = h_final.T @ dlogits
        dby = np.sum(dlogits, axis = 0)
        dh_next = dlogits @ self.Wyh.T

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(window_length)):
            dtanh = (1 - hs[t] ** 2) * dh_next
            dbh += np.sum(dtanh, axis = 0)
            dWxh +=  X[:,t].T @ dtanh
            dWhh += hs[t-1].T @ dtanh
            dh_next =  dtanh @ self.Whh.T

        gradients = {
            "dWxh": dWxh,
            "dWhh": dWhh,
            "dWyh": dWyh,
            "dby": dby,
            "dbh": dbh
        }

        return gradients
        


    
    def update_params(self, gradients, max_grad_norm):
        for key in gradients:
            gradients[key] = np.clip(gradients[key], -max_grad_norm, max_grad_norm)

        self.Whh -= self.learning_rate * gradients["dWhh"]
        self.Wxh -= self.learning_rate * gradients["dWxh"]
        self.Wyh -= self.learning_rate * gradients["dWyh"]
        self.bh -= self.learning_rate * gradients["dbh"]
        self.by -= self.learning_rate * gradients["dby"]
        



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

    # mapping activities to integers
    dataset['activity'] = dataset['activity'].map(ACTIVITY_MAP)

    # print(dataset.head())

    train_dataset = dataset[(dataset['user']) <= 25].copy()
    test_dataset = dataset[dataset['user'] > 25].copy()

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
    print(X_train.shape)
    

    
    

if __name__ == "__main__":
    main()