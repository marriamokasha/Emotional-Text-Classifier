# 💗 SentimentScope: Emotional Text Classifier

> An AI-powered dashboard that reads between the lines and understands human emotions in text

## ✨ What It Does

SentimentScope transforms raw text into emotional insights using machine learning. Whether you're analyzing customer feedback, social media posts, or personal messages, this tool identifies the underlying emotions with impressive accuracy.

## 🎯 Key Features

- **6 Emotion Detection**: Anger, Fear, Joy, Love, Sadness, Surprise
- **Interactive Dashboard**: Beautiful visualizations powered by Plotly and Dash
- **Multiple ML Models**: Logistic Regression, XGBoost comparison
- **Real-time Predictions**: Test your own text instantly
- **Word Intelligence**: See which words drive each emotion
- **Visual Analytics**: Confusion matrices, word clouds, and performance metrics

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/marriamokasha/Emotional-Text-Classifier
   cd Emotional-Text-Classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   ```bash
   python split_data.py
   python prepare_visual_data.py
   python generate_wordclouds.py
   ```

4. **Train the models**
   ```bash
   jupyter notebook train_model.ipynb
   ```

5. **Launch the dashboard**
   ```bash
   python app.py
   ```

6. **Open your browser** → `http://localhost:8060`

## 🎨 What You'll See

- **📊 Overview**: Dataset statistics and emotion distribution
- **🔍 Data Explorer**: Interactive charts and detailed breakdowns  
- **☁️ Word Intelligence**: Word clouds and frequency analysis
- **🏆 Model Arena**: Performance comparison and confusion matrices
- **🤖 Live Predictor**: Test emotions on your own text in real-time

## 🧠 Models & Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 89.6% | 89.5% |
| XGBoost | 89.6% | 89.8% |

## 📁 Project Structure

```
📦 Emotional-Text-Classifier
├── 🎨 app.py                 # Main dashboard application
├── 📊 data/                  # Dataset files
├── 🤖 model/                 # Trained models & results
├── ☁️ wordclouds/           # Generated word clouds
├── 📈 processed/            # Preprocessed data
└── 🔧 *.py                  # Data processing scripts
```

## 🎯 Use Cases

- **Customer Service**: Analyze support ticket emotions
- **Social Media**: Monitor brand sentiment
- **Content Creation**: Understand audience emotional response
- **Research**: Study emotional patterns in text data
- **Personal**: Analyze your own writing emotional tone

## 🛠️ Tech Stack

- **Backend**: Python, Scikit-learn, XGBoost, Pandas
- **Frontend**: Dash, Plotly, Bootstrap
- **Visualization**: WordCloud, Matplotlib
- **Models**: TF-IDF + Logistic Regression/XGBoost

## 📊 Dataset

The model is trained on emotional text data with 6 distinct emotion categories, providing balanced and accurate emotion classification across different text types.

---

**Ready to dive into the emotional landscape of text? Fire up SentimentScope and start discovering the feelings hidden in words! 🚀**