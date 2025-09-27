from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(message: str) -> dict:
    return analyzer.polarity_scores(message)


if __name__=="__main__":
    message="Nothing is wrong"
    print(message)
    print(analyze_sentiment(message=message))