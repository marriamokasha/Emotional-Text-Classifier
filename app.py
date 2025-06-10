# app.py - Enhanced Version
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import joblib
import os
import numpy as np
from wordcloud import WordCloud
import base64
from datetime import datetime

# === Load Data ===
emotion_counts = pd.read_csv('processed/emotion_counts.csv')
top_words = pd.read_json('processed/top_words_per_emotion.json')
model_results = pd.read_csv('model/model_results.csv')
label_encoder = joblib.load('model/label_encoder.pkl')
wordcloud_folder = 'wordclouds'

# === Load Confusion Matrices ===
confusion_matrices = {}
for model_name in model_results['model']:
    
    csv_path = f"model/{model_name}/confusion_matrix.csv"
    cm_df = pd.read_csv(csv_path, index_col=0)
    confusion_matrices[model_name] = cm_df.values


# === Load Trained Models ===
models = {}
for _, row in model_results.iterrows():
    models[row['model']] = joblib.load(row['path'])

# === Enhanced Helper Functions ===
emoji_map = {
    'anger': '😠', 'fear': '😨', 'joy': '😄', 
    'love': '😍', 'sadness': '😢', 'surprise': '😲'
}

color_map = {
    'anger': "#D60101", 'fear': '#4ECDC4', 'joy': '#FFE66D',
    'love': '#FF8E9B', 'sadness': '#6C5CE7', 'surprise': "#79FD7D"
}

def create_enhanced_word_comparison():
    """Create comparative word analysis across all emotions"""
    emotion_word_data = []
    
    for emotion in top_words.keys():
        # Handle both list and dict formats
        if isinstance(top_words[emotion], dict):
            words = top_words[emotion].get('words', [])[:10]
            counts = top_words[emotion].get('counts', [])[:10]
        else:
            # If it's a list of tuples
            words = [item[0] for item in top_words[emotion][:10]]
            counts = [item[1] for item in top_words[emotion][:10]]
        
        for word, count in zip(words, counts):
            emotion_word_data.append({
                'emotion': emotion,
                'word': word,
                'count': count,
                'emoji': emoji_map.get(emotion, '❓')
            })
    
    return pd.DataFrame(emotion_word_data)

def create_model_comparison_data():
    """Prepare data for comprehensive model comparison"""
    comparison_data = []
    for _, row in model_results.iterrows():
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            comparison_data.append({
                'model': row['model'],
                'metric': metric.replace('_', ' ').title(),
                'value': row[metric],
                'percentage': row[metric] * 100
            })
    return pd.DataFrame(comparison_data)

# === App Initialization ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)
app.title = "SentimentScope"

# === Enhanced Layout ===
app.layout = dbc.Container([
    # Header with gradient background
    html.Div([
        html.H1("💗 SentimentScope: Emotional Text Classifier", 
                className="text-center text-white mb-0 py-4"),
        html.P("Advanced ML Dashboard for Emotion Analysis", 
                className="text-center text-white-50 mb-0")
    ], style={'background': 'linear-gradient(45deg, #667eea 0%, #764ba2 100%)', 
                'margin': '-15px -15px 30px -15px', 'padding': '20px'}),
    
    # Enhanced Navigation
    dcc.Tabs(id="tabs", value='home', className="mb-4", children=[
        dcc.Tab(label="🏠 Overview", value='home', className="custom-tab"),
        dcc.Tab(label="📊 Data Explorer", value='exploration', className="custom-tab"),
        dcc.Tab(label="🔍 Word Intelligence", value='words', className="custom-tab"),
        dcc.Tab(label="🏆 Model Arena", value='performance', className="custom-tab"),
        dcc.Tab(label="🤖 Live Predictor", value='live', className="custom-tab")
    ]),
    
    html.Div(id='tabs-content'),
    
    # Footer
    html.Hr(),
    html.P("🎓 Data Visualization Final Project | Emotional Text Analysis Dashboard", 
           className="text-center text-muted small")
], fluid=True)

# === Enhanced Callbacks ===
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'home':
        return create_home_tab()
    elif tab == 'exploration':
        return create_exploration_tab()
    elif tab == 'words':
        return create_words_tab()
    elif tab == 'performance':
        return create_performance_tab()
    elif tab == 'live':
        return create_live_tab()

def create_home_tab():
    """Enhanced home tab with key insights"""
    total_samples = emotion_counts['count'].sum()
    dominant_emotion = emotion_counts.loc[emotion_counts['count'].idxmax(), 'emotion']
    best_model = model_results.loc[model_results['accuracy'].idxmax(), 'model']
    best_accuracy = model_results['accuracy'].max()
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("📊 Dataset Overview", className="card-title"),
                        html.H2(f"{total_samples:,}", className="text-primary"),
                        html.P("Total Text Samples", className="text-muted")
                    ])
                ], color="light")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("🎯 Dominant Emotion", className="card-title"),
                        html.H2(f"{emoji_map[dominant_emotion]} {dominant_emotion.title()}", className="text-warning"),
                        html.P(f"{emotion_counts.loc[emotion_counts['emotion']==dominant_emotion, 'count'].iloc[0]} samples", className="text-muted")
                    ])
                ], color="light")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("🏆 Best Model", className="card-title"),
                        html.H2(best_model, className="text-success"),
                        html.P(f"{best_accuracy:.1%} Accuracy", className="text-muted")
                    ])
                ], color="light")
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("🎨 Emotions", className="card-title"),
                        html.H2(len(emotion_counts), className="text-info"),
                        html.P("Different Categories", className="text-muted")
                    ])
                ], color="light")
            ], md=3)
        ], className="mb-4"),
        
        # Emotion showcase
        html.Div([
            html.H3("Emotion Spectrum", className="text-center mb-4"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1(emoji_map[emotion], className="mb-2"),
                        html.H5(emotion.title(), className="mb-1"),
                        html.P(f"{row['count']} samples", className="text-muted small")
                    ], className="text-center p-3 rounded", 
                    style={'backgroundColor': color_map[emotion] + '20', 'border': f'2px solid {color_map[emotion]}'})
                ], md=2) for emotion, row in emotion_counts.set_index('emotion').iterrows()
            ], justify="center")
        ])
    ])

def create_exploration_tab():
    """Enhanced data exploration with interactive charts"""
    # Enhanced pie chart
    pie_fig = px.pie(emotion_counts, names='emotion', values='count', 
                     title="📊 Emotion Distribution", hole=0.4,
                     color='emotion', color_discrete_map=color_map)
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(showlegend=False, height=400)
    
    # Enhanced bar chart with better styling
    bar_fig = px.bar(emotion_counts, x='emotion', y='count', 
                     title="📈 Sample Count by Emotion",
                     color='emotion', color_discrete_map=color_map)
    bar_fig.update_layout(showlegend=False, xaxis_title="Emotion", yaxis_title="Number of Samples")
    bar_fig.update_traces(texttemplate='%{y}', textposition='inside')
    
    # Distribution analysis
    stats_data = []
    for _, row in emotion_counts.iterrows():
        percentage = (row['count'] / emotion_counts['count'].sum()) * 100
        stats_data.append({
            'emotion': row['emotion'].title(),
            'count': row['count'],
            'percentage': f"{percentage:.1f}%",
            'emoji': emoji_map[row['emotion']]
        })
    
    return html.Div([
        # dbc.Row([
        #     dbc.Col([dcc.Graph(figure=pie_fig)], md=6),
        #     dbc.Col([dcc.Graph(figure=bar_fig)], md=6)
        # ]),
        
        html.Hr(),
        html.H4("📋 Detailed Statistics", className="mt-4 mb-3"),
        
        dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Emotion"), html.Th("Count"), 
                    html.Th("Percentage"), html.Th("Visual")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td([stat['emoji'], " ", stat['emotion']]),
                    html.Td(f"{stat['count']:,}"),
                    html.Td(stat['percentage']),
                    html.Td(html.Div(style={
                        'width': f"{float(stat['percentage'][:-1]) * 3}px",
                        'height': '20px',
                        'backgroundColor': color_map[stat['emotion'].lower()],
                        'borderRadius': '10px'
                    }))
                ]) for stat in stats_data
            ])
        ], striped=True, bordered=True, hover=True)
    ])

def create_words_tab():
    """Enhanced word analysis with comparative insights"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("🎯 Select Emotion for Analysis"),
                dcc.Dropdown(
                    id='emotion-dropdown',
                    options=[{'label': f"{emoji_map[emo]} {emo.title()}", 'value': emo} 
                            for emo in emotion_counts['emotion']],
                    value='joy',
                    clearable=False,
                    style={'marginBottom': '20px'}
                )
            ], md=6),
            dbc.Col([
                html.H4("📊 Analysis Mode"),
                dcc.RadioItems(
                    id='word-analysis-mode',
                    options=[
                        {'label': ' Individual Analysis', 'value': 'individual'},
                        {'label': ' Comparative Analysis', 'value': 'comparative'}
                    ],
                    value='individual',
                    inline=True
                )
            ], md=6)
        ], className="mb-4"),
        
        html.Div(id='word-analysis-content')
    ])

@app.callback(
    Output('word-analysis-content', 'children'),
    [Input('emotion-dropdown', 'value'),
     Input('word-analysis-mode', 'value')]
)
def update_word_analysis(selected_emotion, analysis_mode):
    if analysis_mode == 'individual':
        return create_individual_word_analysis(selected_emotion)
    else:
        return create_comparative_word_analysis()

def create_individual_word_analysis(selected_emotion):
    """Individual emotion word analysis"""
    # Word cloud
    wc_path = os.path.join(wordcloud_folder, f"{selected_emotion}.png")
    if os.path.exists(wc_path):
        encoded = base64.b64encode(open(wc_path, 'rb').read()).decode()
        wordcloud_img = html.Img(src=f'data:image/png;base64,{encoded}', 
                                style={'width': '100%', 'maxHeight': '400px', 'objectFit': 'contain'})
    else:
        wordcloud_img = html.P("Word cloud not available")
    
    # Enhanced bar chart - handle both formats
    try:
        if isinstance(top_words[selected_emotion], dict):
            words = top_words[selected_emotion].get('words', [])[:10]
            counts = top_words[selected_emotion].get('counts', [])[:10]
        else:
            # If it's a list of tuples
            words = [item[0] for item in top_words[selected_emotion][:10]]
            counts = [item[1] for item in top_words[selected_emotion][:10]]
        
        words_data = pd.DataFrame({
            'words': words,
            'counts': counts
        })
        
        bar_fig = px.bar(words_data, x='counts', y='words', orientation='h',
                         title=f"🔤 Top Words in {selected_emotion.title()} {emoji_map.get(selected_emotion, '❓')}",
                         color='counts', color_continuous_scale='Viridis')
        bar_fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    except Exception as e:
        bar_fig = px.bar(title=f"Error loading word data: {str(e)}")
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(f"☁️ Word Cloud - {selected_emotion.title()} {emoji_map.get(selected_emotion, '❓')}"),
                dbc.CardBody([wordcloud_img])
            ])
        ], md=6),
        dbc.Col([
            dcc.Graph(figure=bar_fig)
        ], md=6)
    ])

def create_comparative_word_analysis():
    """Comparative word analysis across emotions"""
    try:
        word_df = create_enhanced_word_comparison()
        
        if word_df.empty:
            return html.P("No word data available for comparison")
        
        # Create heatmap of top words across emotions
        pivot_data = word_df.pivot_table(index='word', columns='emotion', values='count', fill_value=0)
        
        heatmap_fig = px.imshow(pivot_data.values, 
                               x=pivot_data.columns, 
                               y=pivot_data.index,
                               color_continuous_scale='Viridis',
                               title="🌡️ Word Frequency Heatmap Across Emotions")
        heatmap_fig.update_layout(height=600)
        
        # Sunburst chart
        sunburst_fig = px.sunburst(word_df, path=['emotion', 'word'], values='count',
                                  title="🌅 Hierarchical Word Distribution",
                                  color='count', color_continuous_scale='Plasma')
        
        return dbc.Row([
            dbc.Col([dcc.Graph(figure=heatmap_fig)], md=6),
            dbc.Col([dcc.Graph(figure=sunburst_fig)], md=6)
        ])
    except Exception as e:
        return html.Div([
            dbc.Alert(f"Error creating comparative analysis: {str(e)}", color="warning"),
            html.P("Please check your data format in top_words_per_emotion.json")
        ])

def create_performance_tab():
    """Enhanced model performance analysis"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("🎯 Select Model for Detailed Analysis"),
                dcc.Dropdown(
                    id='model-select',
                    options=[{'label': f"🤖 {m}", 'value': m} for m in models.keys()],
                    value='Logistic Regression',
                    clearable=False
                )
            ], md=6),
            dbc.Col([
                html.H4("📊 Comparison View"),
                dcc.RadioItems(
                    id='performance-view',
                    options=[
                        {'label': ' Detailed View', 'value': 'detailed'},
                        {'label': ' Comparison View', 'value': 'comparison'}
                    ],
                    value='detailed',
                    inline=True
                )
            ], md=6)
        ], className="mb-4"),
        
        html.Div(id='performance-content')
    ])

@app.callback(
    Output('performance-content', 'children'),
    [Input('model-select', 'value'),
     Input('performance-view', 'value')]
)
def update_performance_content(model_name, view_mode):
    if view_mode == 'detailed':
        return create_detailed_performance(model_name)
    else:
        return create_comparison_performance()

def create_detailed_performance(model_name):
    """Detailed performance analysis for selected model"""
    row = model_results[model_results['model'] == model_name].iloc[0]
    
    # Metrics cards
    metrics_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🎯 Accuracy", className="card-title"),
                    html.H3(f"{row['accuracy']:.1%}", className="text-primary")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("⚖️ Precision", className="card-title"),
                    html.H3(f"{row['precision']:.1%}", className="text-success")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🔍 Recall", className="card-title"),
                    html.H3(f"{row['recall']:.1%}", className="text-warning")
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🏆 F1 Score", className="card-title"),
                    html.H3(f"{row['f1_score']:.1%}", className="text-info")
                ])
            ])
        ], md=3)
    ], className="mb-4")
    
    # Enhanced confusion matrix
    cm = confusion_matrices.get(model_name)
    if cm is not None:
        cm_fig = px.imshow(cm, 
                          x=label_encoder.classes_, 
                          y=label_encoder.classes_,
                          text_auto=True, 
                          color_continuous_scale='Blues',
                          title=f"🧩 Confusion Matrix - {model_name}")
        cm_fig.update_layout(height=500)
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        class_acc_fig = px.bar(x=label_encoder.classes_, y=per_class_acc,
                              title="📊 Per-Class Accuracy",
                              color=per_class_acc, color_continuous_scale='RdYlGn')
        class_acc_fig.update_layout(yaxis_title="Accuracy", xaxis_title="Emotion")
    else:
        cm_fig = px.bar(title=f"Confusion matrix not available for {model_name}")
        class_acc_fig = px.bar(title="Class accuracy not available")
    
    return html.Div([
        metrics_cards,
        dbc.Row([
            dbc.Col([dcc.Graph(figure=cm_fig)], md=6),
            dbc.Col([dcc.Graph(figure=class_acc_fig)], md=6)
        ])
    ])

def create_comparison_performance():
    """Comparative model performance"""
    comparison_df = create_model_comparison_data()
    
    # Grouped bar chart
    grouped_fig = px.bar(comparison_df, x='model', y='percentage', color='metric',
                        title="🏆 Model Performance Comparison",
                        barmode='group')
    grouped_fig.update_layout(yaxis_title="Score (%)", xaxis_title="Model")
    
    # Radar chart comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    radar_fig = go.Figure()
    
    for _, row in model_results.iterrows():
        values = [row[metric] for metric in metrics]
        radar_fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[m.replace('_', ' ').title() for m in metrics],
            fill='toself',
            name=row['model']
        ))
    
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="🎯 Model Performance Radar"
    )
    
    return dbc.Row([
        dbc.Col([dcc.Graph(figure=grouped_fig)], md=6),
        dbc.Col([dcc.Graph(figure=radar_fig)], md=6)
    ])

def create_live_tab():
    """Enhanced live prediction interface"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🤖 AI-Powered Emotion Predictor"),
                    dbc.CardBody([
                        html.H5("Choose Your Model"),
                        dcc.Dropdown(
                            id='live-model',
                            options=[{'label': f"🧠 {m}", 'value': m} for m in models],
                            value='Logistic Regression',
                            className='mb-3'
                        ),
                        html.H5("Enter Your Text"),
                        dcc.Textarea(
                            id='user-text',
                            placeholder='Type your message here... (e.g., "I am so excited about this project!")',
                            style={'width': '100%', 'height': 120, 'resize': 'vertical'},
                            className='mb-3'
                        ),
                        dbc.Button('🔮 Predict Emotion', id='predict-btn', 
                                 color='primary', className='btn-lg w-100')
                    ])
                ])
            ], md=4),
            dbc.Col([
                html.Div(id='prediction-output')
            ], md=8)
        ])
    ])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-btn', 'n_clicks')],
    [State('user-text', 'value'),
     State('live-model', 'value')]
)
def enhanced_live_prediction(n_clicks, text, model_name):
    if not n_clicks or not text or not text.strip():
        return create_prediction_placeholder()
    
    try:
        model = models[model_name]
        probs = model.predict_proba([text.strip()])[0]
        pred_idx = np.argmax(probs)
        emotion = label_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100
        
        return create_enhanced_prediction_display(text, emotion, confidence, probs, model_name)
    
    except Exception as e:
        return dbc.Alert(f"Error in prediction: {str(e)}", color="danger")

def create_prediction_placeholder():
    """Placeholder for prediction output"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-robot fa-3x text-muted mb-3"),
                html.H4("Ready to Analyze!", className="text-muted"),
                html.P("Enter some text and click 'Predict Emotion' to see the magic happen!", 
                       className="text-muted")
            ], className="text-center py-5")
        ])
    ], style={'minHeight': '400px'})

def create_enhanced_prediction_display(text, emotion, confidence, probs, model_name):
    """Enhanced prediction display with detailed analysis"""
    
    # Main prediction card
    main_card = dbc.Card([
        dbc.CardHeader([
            html.H4([
                f"🎯 Predicted Emotion: {emotion.title()} ",
                html.Span(emoji_map[emotion], style={'fontSize': '2rem'})
            ])
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Input Text:"),
                    html.P(f'"{text}"', className="font-italic text-muted border-left pl-3"),
                    html.H5("Confidence Level:"),
                    dbc.Progress(value=confidence, color="success" if confidence > 70 else "warning" if confidence > 50 else "danger",
                               striped=True, animated=True, className="mb-2"),
                    html.P(f"{confidence:.1f}% confident", className="small text-muted")
                ], md=6),
                dbc.Col([
                    html.H5("Model Used:"),
                    dbc.Badge(f"🤖 {model_name}", color="info", className="mb-3"),
                    html.H5("Analysis Time:"),
                    html.P(datetime.now().strftime("%H:%M:%S"), className="text-muted")
                ], md=6)
            ])
        ])
    ], color="light", className="mb-4")
    
    # Detailed probability analysis
    prob_data = pd.DataFrame({
        'emotion': label_encoder.classes_,
        'probability': probs * 100,
        'emoji': [emoji_map[e] for e in label_encoder.classes_]
    }).sort_values('probability', ascending=True)
    
    # Horizontal bar chart for probabilities
    prob_fig = px.bar(prob_data, x='probability', y='emotion', orientation='h',
                     title="🎲 Emotion Probability Breakdown",
                     color='probability', color_continuous_scale='Viridis',
                     text='probability')
    prob_fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    prob_fig.update_layout(height=400, xaxis_title="Probability (%)")
    
    # Radar chart for probabilities
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=probs,
        theta=[f"{e} {emoji_map[e]}" for e in label_encoder.classes_],
        fill='toself',
        line_color=color_map[emotion],
        fillcolor=color_map[emotion],
        opacity=0.6,
        name='Probabilities'
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="🎯 Emotion Confidence Radar",
        showlegend=False,
        height=400
    )
    
    # Additional insights
    top_2_emotions = prob_data.nlargest(2, 'probability')
    second_emotion = top_2_emotions.iloc[1]
    
    insights_card = dbc.Card([
        dbc.CardHeader("🔍 Analysis Insights"),
        dbc.CardBody([
            html.Ul([
                html.Li(f"Primary emotion detected: {emotion.title()} with {confidence:.1f}% confidence"),
                html.Li(f"Secondary emotion: {second_emotion['emotion'].title()} ({second_emotion['probability']:.1f}%)"),
                html.Li(f"Confidence level: {'High' if confidence > 70 else 'Medium' if confidence > 50 else 'Low'}"),
                html.Li(f"Model certainty: {'Very certain' if confidence > 80 else 'Moderately certain' if confidence > 60 else 'Uncertain'}")
            ])
        ])
    ], className="mb-4")
    
    return html.Div([
        main_card,
        insights_card,
        dbc.Row([
            dbc.Col([dcc.Graph(figure=prob_fig)], md=6),
            dbc.Col([dcc.Graph(figure=radar_fig)], md=6)
        ])
    ])

# === Custom CSS ===
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .custom-tab {
                font-weight: bold;
            }
            .custom-tab:hover {
                background-color: #f8f9fa;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .card {
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s;
            }
            .card:hover {
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True, port=8060)