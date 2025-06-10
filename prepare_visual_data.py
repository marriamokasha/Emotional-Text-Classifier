import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
import json


train = pd.read_csv('data/train.csv', names=['text', 'label'])

emotion_counts = train['label'].value_counts().reset_index()
emotion_counts.columns = ['emotion', 'count']

os.makedirs("processed", exist_ok=True)
emotion_counts.to_csv("processed/emotion_counts.csv", index=False)

print("✅ Emotion counts saved to 'processed/emotion_counts.csv'")

vectorizer = CountVectorizer(stop_words='english')
top_words_dict = {}

for emotion in train['label'].unique():
    texts = train[train['label'] == emotion]['text']
    X = vectorizer.fit_transform(texts) 
    vocab = vectorizer.get_feature_names_out()
    top_n = 10
    word_sums = np.array(X.sum(axis=0)).flatten()
    top_indices = word_sums.argsort()[::-1][5:top_n+5]
    top_words = [(vocab[i], int(word_sums[i])) for i in top_indices]
    top_words_dict[emotion] = top_words
    
with open("processed/top_words_per_emotion.json", "w") as f:
    json.dump(top_words_dict, f, indent=2)

print("✅ Top words saved to 'processed/top_words_per_emotion.json'")