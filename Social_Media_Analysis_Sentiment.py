#!/usr/bin/env python
# coding: utf-8

# In[69]:


import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'social-media-sentiments-analysis-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F4245661%2F7316566%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240813%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240813T142532Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D3703d9bf24833da45754e49132255ec16f93976c08d0aea17c41291a44991cbf9ec805f2db0c89f0f902cfa574d3f29ac8c16541fc6d6de72ccd63c3fba46296e83317efa43b1d5591034b3b2a8824769ef81fa79824c61530443d05b5601f2c5e5524557fbfcdfa28ff2d6e33fece60601c1da3b4761885cf66681601cf369a99d1c8db69889227bd0a7b9f8246c35fa164ff02a4836b29ad0542d0dac54d5d4f9f0785611d8be3bd6cc213beb5b1a21684d657f1211a562fef9bd54c9b9706353d6db64004275c69b63aa3073572ac9deb6989641d3ad101b19991dc8010d559b19eb1bbec4b12d0442d0c6863120440ffe5f42bf8afe526c6f1cd461bdb31'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

get_ipython().system('umount /kaggle/input/ 2> /dev/null')
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 15px; font-weight: bold;">
#         1. Import Libraries
#    </p>
# </div>
# 

# In[70]:


pip install twython


# In[71]:


pip install vaderSentiment


# In[72]:


pip install colorama


# In[73]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init
import plotly.express as px

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm
from collections import Counter
from wordcloud import WordCloud




nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

import warnings
warnings.filterwarnings('ignore')


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 15px; font-weight: bold;">
#         3. Load Data
#    </p>
# </div>
# 

# In[74]:


df = pd.read_csv("/kaggle/input/social-media-sentiments-analysis-dataset/sentimentdataset.csv")


# In[75]:


df.head()


# In[76]:


def null_count():
    return pd.DataFrame({'features': df.columns,
                'dtypes': df.dtypes.values,
                'NaN count': df.isnull().sum().values,
                'NaN percentage': df.isnull().sum().values/df.shape[0]}).style.background_gradient(cmap='Set3',low=0.1,high=0.01)
null_count()


# In[77]:


df.duplicated().sum()


# In[78]:


df.columns


# In[79]:


for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 15px; font-weight: bold;">
#         3. Feature Enginering
#    </p>
# </div>
# 

# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#         Drop Columns
#    </p>
# </div>
# 

# In[80]:


df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'Hashtags','Day', 'Hour','Sentiment'])


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Platform
#    </p>
# </div>
# 

# In[81]:


df['Platform'].value_counts()


# In[82]:


df['Platform'] = df['Platform'].str.strip()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Country
#    </p>
# </div>
# 

# In[83]:


df['Country'].value_counts()


# In[84]:


df['Country'] = df['Country'].str.strip()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Timestamp
#    </p>
# </div>
# 

# In[85]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Day_of_Week'] = df['Timestamp'].dt.day_name()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Month
#    </p>
# </div>
# 

# In[86]:


month_mapping = {
    1: 'Januari',
    2: 'Februari',
    3: 'Maret',
    4: 'April',
    5: 'Mei',
    6: 'Juni',
    7: 'Juli',
    8: 'Agustus',
    9: 'September',
    10: 'Oktober',
    11: 'November',
    12: 'Desember'
}

df['Month'] = df['Month'].map(month_mapping)

df['Month'] = df['Month'].astype('object')


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Text
#    </p>
# </div>
# 

# In[87]:


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = " ".join(text.split())
    tokens = word_tokenize(text)

    cleaned_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]

    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

df["Clean_Text"] = df["Text"].apply(clean)


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Unique Columns
#    </p>
# </div>
# 

# In[88]:


specified_columns = ['Platform','Country', 'Year','Month','Day_of_Week']

for col in specified_columns:
    total_unique_values = df[col].nunique()
    print(f'Total unique values for {col}: {total_unique_values}')

    top_values = df[col].value_counts()

    colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE, Fore.LIGHTBLACK_EX, Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX]

    for i, (value, count) in enumerate(top_values.items()):
        color = colors[i % len(colors)]
        print(f'{color}{value}: {count}{Fore.RESET}')

    print('\n' + '=' * 30 + '\n')


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 15px; font-weight: bold;">
#         4. E D A
#    </p>
# </div>
# 

# In[89]:


df1 = df.copy()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 15px; font-weight: bold;">
#        4.1 Sentiment Analysis
#    </p>
# </div>
# 

# ![Arabic-Sentiment-Analysis-2.jpg](attachment:1d3df7bf-bc5e-4d59-880d-8f2189fb48a0.jpg)

# In[90]:


analyzer = SentimentIntensityAnalyzer()

df1['Vader_Score'] = df1['Clean_Text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

df1['Sentiment'] = df1['Vader_Score'].apply(lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral'))

print(df1[['Clean_Text', 'Vader_Score', 'Sentiment']].head())


# In[91]:


colors = ['#66b3ff', '#99ff99', '#ffcc99']

explode = (0.1, 0, 0)

sentiment_counts = df1.groupby("Sentiment").size()

fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(
    x=sentiment_counts,
    labels=sentiment_counts.index,
    autopct=lambda p: f'{p:.2f}%\n({int(p*sum(sentiment_counts)/100)})',
    wedgeprops=dict(width=0.7),
    textprops=dict(size=10, color="r"),
    pctdistance=0.7,
    colors=colors,
    explode=explode,
    shadow=True)

center_circle = plt.Circle((0, 0), 0.6, color='white', fc='white', linewidth=1.25)
fig.gca().add_artist(center_circle)

ax.text(0, 0, 'Sentiment\nDistribution', ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

ax.legend(sentiment_counts.index, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

ax.axis('equal')

plt.show()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Year
#    </p>
# </div>
# 

# In[92]:


plt.figure(figsize=(12, 6))
sns.countplot(x='Year', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Years and Sentiment')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Month
#    </p>
# </div>
# 

# In[93]:


plt.figure(figsize=(12, 6))
sns.countplot(x='Month', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Month and Sentiment')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Day Of Weeek
#    </p>
# </div>
# 

# In[94]:


plt.figure(figsize=(12, 6))
sns.countplot(x='Day_of_Week', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Day of Week and Sentiment')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Platform
#    </p>
# </div>
# 

# In[95]:


plt.figure(figsize=(12, 6))
sns.countplot(x='Platform', hue='Sentiment', data=df1, palette='Paired')
plt.title('Relationship between Platform and Sentiment')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Country
#    </p>
# </div>
# 

# In[96]:


plt.figure(figsize=(12, 6))

top_10_countries = df1['Country'].value_counts().head(10).index

df_top_10_countries = df1[df1['Country'].isin(top_10_countries)]

sns.countplot(x='Country', hue='Sentiment', data=df_top_10_countries, palette='Paired')
plt.title('Relationship between Country and Sentiment (Top 10 Countries)')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 15px; font-weight: bold;">
#        4.2 Common Words
#    </p>
# </div>
# 

# ![Transformasi-Teknologi-Informasi-1.jpg](attachment:b2c17ea3-749b-42aa-a17a-17ad84e24925.jpg)

# In[97]:


df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])
top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])

top_words_df.style.background_gradient(cmap='Blues')


# In[98]:


df1['temp_list'] = df1['Clean_Text'].apply(lambda x: str(x).split())
top_words = Counter([item for sublist in df1['temp_list'] for item in sublist])
top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])

fig = px.bar(top_words_df,
            x="count",
            y="Common_words",
            title='Common Words in Text Data',
            orientation='h',
            width=700,
            height=700,
            color='Common_words')

fig.show()


# In[99]:


Positive_sent = df1[df1['Sentiment'] == 'positive']
Negative_sent = df1[df1['Sentiment'] == 'negative']
Neutral_sent = df1[df1['Sentiment'] == 'neutral']


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Positive Common Words
#    </p>
# </div>
# 

# In[100]:


top = Counter([item for sublist in df1[df1['Sentiment'] == 'positive']['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
temp_positive.style.background_gradient(cmap='Greens')


# In[101]:


words = ' '.join([item for sublist in df1[df1['Sentiment'] == 'positive']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Neutral Common Words
#    </p>
# </div>
# 

# In[102]:


top = Counter([item for sublist in df1[df1['Sentiment'] == 'neutral']['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
temp_positive.style.background_gradient(cmap='Blues')


# In[103]:


words = ' '.join([item for sublist in df1[df1['Sentiment'] == 'neutral']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#          Negative Common Words
#    </p>
# </div>
# 

# In[104]:


top = Counter([item for sublist in df1[df1['Sentiment'] == 'negative']['temp_list'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
temp_positive.style.background_gradient(cmap='Reds')


# In[105]:


words = ' '.join([item for sublist in df1[df1['Sentiment'] == 'negative']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 15px; font-weight: bold;">
#         5. Data Preparation
#    </p>
# </div>
# 

# In[106]:


df2 = df1.copy()


# In[107]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#         5.1 Split Data
#    </p>
# </div>
# 

# In[108]:


X = df2['Clean_Text'].values
y = df2['Sentiment'].values


# In[109]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 15px; font-weight: bold;">
#         6. Modeling
#    </p>
# </div>
# 

# In[110]:


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Passive Aggressive Classifier
#    </p>
# </div>
# 

# In[111]:


pac_classifier = PassiveAggressiveClassifier(max_iter=50, random_state=42)
pac_classifier.fit(X_train_tfidf, y_train)


# In[112]:


y_pred = pac_classifier.predict(X_test_tfidf)
accuracy_test = accuracy_score(y_test, y_pred)
classification_rep_test = classification_report(y_test, y_pred)


# In[113]:


print("Test Set Results:")
print(f"Accuracy: {accuracy_test}")
print("Classification Report:\n", classification_rep_test)


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#        Logistic Classifier
#    </p>
# </div>
# 

# In[114]:


logistic_classifier = LogisticRegression(max_iter=50, random_state=42)
logistic_classifier.fit(X_train_tfidf, y_train)


# In[115]:


y_pred_logistic = logistic_classifier.predict(X_test_tfidf)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)


# In[116]:


print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_logistic}")
print("Classification Report:\n", classification_rep_logistic)


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#      Random Fores Classifier
#    </p>
# </div>
# 

# In[117]:


random_forest_classifier = RandomForestClassifier(random_state=42)
random_forest_classifier.fit(X_train_tfidf, y_train)


# In[118]:


y_pred_rf = random_forest_classifier.predict(X_test_tfidf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)


# In[119]:


print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_rf}")
print("Classification Report:\n", classification_rep_rf)


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#      SVM Classifier
#    </p>
# </div>
# 

# In[120]:


svm_classifier = SVC(random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)


# In[121]:


y_pred_svm = svm_classifier.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
classification_rep_svm = classification_report(y_test, y_pred_svm)


# In[122]:


print("Support Vector Machine Results:")
print(f"Accuracy: {accuracy_svm}")
print("Classification Report:\n", classification_rep_svm)


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#      Multinomial NB
#    </p>
# </div>
# 

# In[123]:


nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)


# In[124]:


y_pred_nb = nb_classifier.predict(X_test_tfidf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
classification_rep_nb = classification_report(y_test, y_pred_nb)


# In[125]:


print("\nMultinomial Naive Bayes Results:")
print(f"Accuracy: {accuracy_nb}")
print("Classification Report:\n", classification_rep_nb)


# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#         Best Modeling : Passive Aggressive Classifier
#    </p>
# </div>
# 
# 

# <div style="background-color: purple; padding: 15px; border-radius: 10px;">
#     <p style="color: white; font-size: 12px; font-weight: bold;">
#         Hyperparameters
#    </p>
# </div>
# 
# 

# In[126]:


param_dist = {
    'C': [0.1, 0.5, 1.0],
    'fit_intercept': [True, False],
    'shuffle': [True, False],
    'verbose': [0, 1],
}


# In[127]:


pac_classifier = PassiveAggressiveClassifier(random_state=42)

randomized_search = RandomizedSearchCV(pac_classifier, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
randomized_search.fit(X_train_tfidf, y_train)


# In[128]:


best_params_randomized = randomized_search.best_params_
best_params_randomized


# In[129]:


best_pac_classifier_randomized = PassiveAggressiveClassifier(random_state=42, **best_params_randomized)
best_pac_classifier_randomized.fit(X_train_tfidf, y_train)


# In[130]:


y_pred_best_pac_randomized = best_pac_classifier_randomized.predict(X_test_tfidf)


# In[131]:


accuracy_best_pac_randomized = accuracy_score(y_test, y_pred_best_pac_randomized)
classification_rep_best_pac_randomized = classification_report(y_test, y_pred_best_pac_randomized)
conf_matrix_test = confusion_matrix(y_test, y_pred_best_pac_randomized)


# In[132]:


print("Best PassiveAggressiveClassifier Model (RandomizedSearchCV):")
print(f"Best Hyperparameters: {best_params_randomized}")
print(f"Accuracy: {accuracy_best_pac_randomized}")
print("Classification Report:\n", classification_rep_best_pac_randomized)


# In[133]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Greys', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix - Hyperparameters')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

