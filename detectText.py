import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon', quiet = True)

def analyzeSentiment(text, threshold = 0.05):
    """
    Analyzes the sentiment of a given text using the VADER sentiment analysis tool.

    Args:
        text (str): The input text to analyze for sentiment.
        threshold (float): A value between -1 and 1 used to determine the sentiment polarity.

    Returns:
        str: The sentiment label ("Positive," "Negative," or "Neutral").
        str: The confidence score (in percentage) associated with the sentiment analysis.

    This function utilizes the VADER sentiment analyzer to assess the sentiment of the input text.
    The `threshold` parameter is used to determine the boundary for categorizing sentiment as Positive, Negative, or Neutral.

    If the compound sentiment score is greater than or equal to `threshold`, the sentiment is "Positive."
    If the compound sentiment score is less than or equal to the negative of `threshold`, the sentiment is "Negative."
    Otherwise, the sentiment is "Neutral."

    The confidence is calculated as a percentage of the compound score and is returned as a string. If the sentiment is Neutral, the confidence is "N/A."
    """

    # Get the compound sentiment score using VADER
    compoundScore = SentimentIntensityAnalyzer().polarity_scores(text)['compound']

    # Determine the sentiment and format the confidence score
    if compoundScore >= threshold:
        sentiment = "positive"
        confidence = str(round(compoundScore * 100, 2)) + '%'

    elif compoundScore <= -threshold:
        sentiment = "negative"
        confidence = str(round(compoundScore * -100, 2)) + '%'

    else:
        sentiment = "neutral"
        confidence = "N/A"

    return sentiment, confidence

if __name__ == "__main__":
    while True:
        # Get user input and analyze sentiment
        text = input("Enter a comment (or type 'quit' to close the program): ")

        # Exit the loop if the user types 'exit'
        if text.lower() == 'quit':
            break

        outputSentiment, outputConfidence = analyzeSentiment(text)

        # Print the sentiment and confidence
        print(f"It's a {outputSentiment} sentiment, with {outputConfidence} confidence.\n")