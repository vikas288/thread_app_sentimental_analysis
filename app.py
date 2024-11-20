import os
from io import BytesIO
import base64
from flask import Flask, render_template, request
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk

# Download NLTK data once
nltk.download('vader_lexicon')
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        df = pd.read_csv('threads_reviews.csv')
        df = df.drop(columns=['source'])


        sia = SentimentIntensityAnalyzer()
        df['compound'] = df['review_description'].apply(lambda x: sia.polarity_scores(x)['compound'])


        def sentiment(compound_score):
            if compound_score >= 0.05:
                return 'positive'
            elif -0.05 < compound_score < 0.05:
                return 'neutral'
            else:
                return 'negative'

        df['sentiment'] = df['compound'].apply(sentiment)


        corpus = []
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        for review in df['review_description']:
            review = re.sub('[^a-zA-Z]', ' ', review).lower()
            review = ' '.join([ps.stem(word) for word in review.split() if word not in stop_words])
            corpus.append(review)

        df['cleaned_review'] = corpus


        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df['cleaned_review']).toarray()


        img1 = BytesIO()
        plt.figure(figsize=(10, 5))
        sns.countplot(x='rating', data=df, palette='tab10')
        plt.title("Threads' Rating Counts")
        plt.savefig(img1, format='png')
        plt.close()
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode()


        img2 = BytesIO()
        plt.figure(figsize=(10, 5))
        sns.countplot(x='sentiment', data=df, palette='tab10')
        plt.title("Threads' Sentiment Counts")
        plt.savefig(img2, format='png')
        plt.close()
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()


        img3 = BytesIO()
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['cleaned_review']))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img3, format='png')
        plt.close()
        img3.seek(0)
        plot_url3 = base64.b64encode(img3.getvalue()).decode()


        img4 = BytesIO()
        df['review_date'] = pd.to_datetime(df['review_date'])
        reviewbydate = df.groupby('review_date')['compound'].sum()

        plt.figure(figsize=(11, 4))
        plt.plot(reviewbydate.index, reviewbydate.values, label='Sentiment', color='peru')
        plt.axhline(y=0, color='maroon', linestyle='solid', label='Threshold (0)')
        plt.title("Threads' Sentiment Distribution by Date")
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(img4, format='png')
        plt.close()
        img4.seek(0)
        plot_url4 = base64.b64encode(img4.getvalue()).decode()


        positive_reviews = df[df['sentiment'] == 'positive'].shape[0]
        neutral_reviews = df[df['sentiment'] == 'neutral'].shape[0]
        negative_reviews = df[df['sentiment'] == 'negative'].shape[0]

        return render_template(
            'result.html',
            positive=positive_reviews,
            neutral=neutral_reviews,
            negative=negative_reviews,
            plot_url1=plot_url1,
            plot_url2=plot_url2,
            plot_url3=plot_url3,
            plot_url4=plot_url4
        )

if __name__ == '__main__':
    app.run(debug=True)
