import pandas as pd
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import os


train = pd.read_csv('data/train.csv', names=['text', 'label'])


os.makedirs('wordclouds', exist_ok=True)

color_schemes = {
    'anger': 'Reds',
    'fear': 'Blues', 
    'joy': 'YlOrRd',
    'love': 'RdPu',
    'sadness': 'BuPu',
    'surprise': 'Greens'
}

custom_stopwords = {'feeling', 'feelings', 'like', 'im', 'just', 'feel'}
all_stopwords = STOPWORDS.union(custom_stopwords)


for emotion in train['label'].unique():
    texts = train[train['label'] == emotion]['text']
    full_text = " ".join(texts)

    colormap = color_schemes.get(emotion, 'Set2')
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        colormap=colormap,
        stopwords=all_stopwords,
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(full_text)
    
    
    file_path = os.path.join('wordclouds', f"{emotion}.png")
    wc.to_file(file_path)
    print(f"✅ Saved: {file_path}")
