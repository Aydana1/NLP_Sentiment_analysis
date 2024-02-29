
import random 
import tabula
from tabulate import tabulate
import pandas as pd
import spacy
import re
import nltk
# nltk.download('all')
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from translate import Translator

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from detoxify import Detoxify
from spacy_langdetect import LanguageDetector
from spacy.language import Language

class PDF_doc:
    pdf_path = ''

    def read_pdf(self, pdf_path):
        tables = tabula.read_pdf(pdf_path, pages='1', multiple_tables=False) #area=[100.51, 21.51, 564.62, 595.92]
        df_tables = pd.DataFrame((tables[0]))
        return df_tables
    
    def df_to_list(self, str_indx, df_tables):

        str_arr = []
        dates_arr = []
        time_arr = []
        gender_arr = []

        for j in range(len(str_indx)-1):
            str = ''
            for i in range(str_indx[j], str_indx[j+1]):
                if i == str_indx[len(str_indx)-2]:
                    str += df_tables['Answer'][i] + df_tables['Answer'][i+1] + df_tables['Answer'][i+2]
                    break
                str += df_tables['Answer'][i] + ' '
                
            date = df_tables['Date'][str_indx[j]]
            if date == '01/01/2023':
                continue
            str_arr.append(str)
            time = df_tables['Time'][str_indx[j]]
            gender = df_tables['Male / Female'][str_indx[j]]
            dates_arr.append(date)
            time_arr.append(time)
            gender_arr.append(gender)
        
        return str_arr, dates_arr, time_arr, gender_arr
    

    def extract_info(self, conversation):
        # Load spaCy's English NLP model
        nlp = spacy.load("en_core_web_sm")
        # Process the conversation using spaCy
        doc = nlp(conversation)

        # Extract person names and locations
        person_names = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        person_locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']  # GPE stands for geopolitical entity

        # Extract ages using regular expression
        ages = re.findall(r'\b(\d{1,2})\s?(?:years old|y.o.)\b', conversation)
    
        return person_names, person_locations, ages
    

    # create preprocess_text function: removes stop words like articles, lemitizes, and tokenization. You can use it before sentiment analysis to prepare the text.
    def preprocess_text(self, text):
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        # Remove stop words
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        # Join the tokens back into a string
        processed_text = ' '.join(lemmatized_tokens)

        return processed_text
    
    def detect_lang(self, text):
        result_lang = detect(text)
        return result_lang 
    
    def translate_text(self, result_lang, text):
        random.seed(0)
        translator= Translator(from_lang=result_lang, to_lang="en")
        translation = translator.translate(text)
        return translation
    
    def text_blob_sentiment(self, text):
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        sentiment_blob = blob.sentiment.polarity  # Range from -1 to 1
        return sentiment_blob

    def vader_sentiment(self, text):
        # Vader sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        sentiment_vader = analyzer.polarity_scores(text)['compound']  # Range from -1 to 1
        if(sentiment_vader < 0):  #if negative score, normalize between (0, 1)
            sentiment_vader = (sentiment_vader + 1)/2
        return sentiment_vader

    def detect_rudeness(self, text):
        rudeness_score = Detoxify('original').predict(text)
        return rudeness_score
    
    def process_pdf(self, input, i):
        if i == 3:
            result_lang = 'fr'  
        else:
            result_lang = self.detect_lang(input)
        if result_lang != "en":
            translation = self.translate_text(result_lang, input)
        else:
            translation = input
        print(translation)
        person_names, person_locations, ages = self.extract_info(translation)
        blob_score = self.text_blob_sentiment(translation)
        vader_score = self.vader_sentiment(translation)
        rudeness_score = self.detect_rudeness(translation)
        
        return translation, person_names, person_locations, ages, blob_score, vader_score, rudeness_score
