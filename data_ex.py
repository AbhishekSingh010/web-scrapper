import pandas as pd
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import syllables

# Set NLTK data path
nltk.data.path.append(r'C:\Users\abhis\AppData\Roaming\nltk_data\sentiment\vader_lexicon.zip')

# Download required NLTK resources
nltk.download('punkt')

# Load Stop Words
stop_word_files = ['StopWords_Auditor.txt', 'StopWords_Currencies.txt', 'StopWords_DatesandNumbers.txt', 'StopWords_Generic.txt', 'StopWords_GenericLong.txt', 'StopWords_Geographic.txt', 'StopWords_Names.txt']

stop_words = []
for file_name in stop_word_files:
    with open(file_name, 'r') as file:
        stop_words.extend(line.strip() for line in file)

# Load Positive and Negative Words
positive_words = []
with open('positive-words.txt', 'r') as file:
    positive_words.extend(line.strip() for line in file)

negative_words = []
with open('negative-words.txt', 'r') as file:
    negative_words.extend(line.strip() for line in file)

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def is_complex(word):
    return len(word) >= 8

def calculate_total_syllables(words):
    return sum(syllables.estimate(word) for word in words)

def analyze_article(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    word_count = len(filtered_tokens)
    total_syllables = calculate_total_syllables(filtered_tokens)
    
    pos_word_count = sum(1 for token in filtered_tokens if token in positive_words)
    neg_word_count = sum(1 for token in filtered_tokens if token in negative_words)
    complex_word_count = sum(1 for token in filtered_tokens if is_complex(token))
    
    sentences = nltk.sent_tokenize(text)
    avg_sentence_length = sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences) if sentences else 0
    avg_words_per_sentence = word_count / len(sentences) if sentences else 0
    complex_word_percentage = (complex_word_count / word_count) * 100 if word_count > 0 else 0
    
    sentiment_scores = sia.polarity_scores(text)
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    polarity_score = sentiment_scores['compound']
    
    syllable_per_word = total_syllables / word_count if word_count > 0 else 0
    
    personal_pronouns = sum(1 for token in tokens if token.lower() in ['i', 'me', 'my', 'mine', 'you', 'your', 'yours'])
    
    total_word_length = sum(len(token) for token in filtered_tokens)
    avg_word_length = total_word_length / word_count if word_count > 0 else 0
    
    fog_index = 0.4 * (avg_words_per_sentence + complex_word_percentage)
    
    return [pos_word_count, neg_word_count, positive_score, negative_score, polarity_score,
            avg_sentence_length, complex_word_percentage, fog_index, avg_words_per_sentence,
            complex_word_count, word_count, syllable_per_word, personal_pronouns, avg_word_length]

input_data = pd.read_excel('Input.xlsx')
output_data = []

for index, row in input_data.iterrows():
    response = requests.get(row['URL'])
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        article_text = soup.find('article').text
        analysis_result = analyze_article(article_text)
        output_data.append([row["URL_ID"],row['URL']] + analysis_result)

output_columns = [
    'URL_ID','URL', 'POSITIVE WORD COUNT', 'NEGATIVE WORD COUNT', 'POSITIVE SCORE', 'NEGATIVE SCORE',
    'POLARITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
    'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD',
    'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
]

output_df = pd.DataFrame(output_data, columns=output_columns)
output_df.to_excel('Output.xlsx', index=False)
