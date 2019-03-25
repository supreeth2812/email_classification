import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(50)
pd.set_option('display.max_columns',10)
pd.set_option('display.width',10)
pd.options.mode.chained_assignment = None

##Import the csv file
df = pd.read_csv('enron_cleaned_sent_emails.csv')

##Code to split the file field and fetch sender information
df_new = df.file.str.split(pat = "/", expand = True)
df_new.columns = ['sender','folder','fileno','none']
df["sender"] = df_new.sender

Top_senders = list(df["sender"].value_counts().head(10).index.values)
df_cleaned = df[df.sender.isin(Top_senders)]

##Plot the top 10 senders values
plt.figure(figsize=(10,4))
df_cleaned["sender"].value_counts().head(20).plot(kind='bar')

##Number of total words in our dataset in Body field
print(df_cleaned['body'].apply(lambda x: len(x.split(' '))).sum())
#10,885,068

##Drop rows with Nan values
df_cleaned.dropna(inplace=True)

df_cleaned["clean_body"] = df.body

##Remove punctuations from text data
df_cleaned.clean_body = df.body.apply(lambda x: x.translate(str.maketrans('','', string.punctuation)))

##Remove digits and tab, newlines, spaces
df_cleaned.clean_body = df_cleaned.clean_body.apply(lambda x: x.translate(str.maketrans('','', string.digits)))
df_cleaned.clean_body = df_cleaned.clean_body.apply(lambda x: x.translate(str.maketrans('','', "\n")))
df_cleaned.clean_body = df_cleaned.clean_body.apply(lambda x: x.translate(str.maketrans('','', "\t")))

##Remove stop words
stop = stop_words = set(stopwords.words('english'))
stop.update(("to","cc","subject","http","from","sent","msn"))

def remove_stopwords(text):
    text = ' '.join(word for word in text.split() if word not in stop)
    return text

df_cleaned.clean_body = df_cleaned.clean_body.apply(remove_stopwords)

##Lemmatizing and Word tokenizing to normalize the text
lemmatizer = WordNetLemmatizer()
w_tokenizer = WhitespaceTokenizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df_cleaned.clean_body = df_cleaned.clean_body.apply(lemmatize_text)
df_cleaned.clean_body = df_cleaned.clean_body.apply(lambda x: TreebankWordDetokenizer().detokenize(x))

##After data cleaning and NLP check total word count in Body field
print(df_cleaned['clean_body'].apply(lambda x: len(x.split(' '))).sum())
#5,156,830

##Transform target variable(10 senders) to categories
df_cleaned["sender"] = df_cleaned["sender"].astype('category')
df_cleaned["sender_cat"] = df_cleaned["sender"].cat.codes

X = df_cleaned.clean_body
y = df_cleaned.sender_cat

##Split data into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

##Naive Bayes Classifier - Rejected
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=Top_senders))

##Linear Support Vector Classifier - Rejected
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
	
	
##Logistic Regression Classifier - Accepted Final tuned model
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver = "lbfgs", C=1e5, n_jobs = 1)),
               ])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=Top_senders))


##Calculate and plot Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=Top_senders, yticklabels=Top_senders)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

