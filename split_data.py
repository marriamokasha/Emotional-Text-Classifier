import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('data/data.csv', names=['text', 'label'])

train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

test.to_csv('data/test.csv', index=False, header=False)
train.to_csv('data/train.csv', index=False, header=False)
