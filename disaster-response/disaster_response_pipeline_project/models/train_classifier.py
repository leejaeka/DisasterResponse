import sys
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.corpus import stopwords
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
	'''
	INPUT:
	database_filepath - path to load
	OUTPUT:
	X - dataframe with all the messages
	y - dataframe with all classification
	cats - all possible classification categories as list
	'''
	engine = create_engine('sqlite:///' + database_filepath)
	conn = engine.connect()
	df = pd.read_sql('SELECT * FROM DisasterResponse', con = conn)
	X = df['message']
	y = df.drop(['id','message','original', 'genre'], axis=1)
	cats = list(y.columns)
	return X, y, cats


def tokenize(text):
	'''
	INPUT:
	text - text to tokenize
	OUTPUT:
	tokens - list of tokenized English words
    '''
	en_stops = set(stopwords.words('english'))
	text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
	sentence = word_tokenize(text)
	tokens = [WordNetLemmatizer().lemmatize(word) for word in sentence if word not in en_stops]
	return tokens


def build_model():
	'''
	OUTPUT:
	Pipeline - NLP and ML pipeline of the model
	'''
	pipeline = Pipeline([
		('features', FeatureUnion([
			('text_pipeline', Pipeline([
				('vect', CountVectorizer(tokenizer=tokenize, max_df=1.0, max_features=None)),
				('tfidf', TfidfTransformer(use_idf=True))
			]))
		])),
		('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200))) 
	], verbose=True)
	# GridSearchCV for REFERENCE ONLY. Note that running this will take a million year
	#parameters = {
    #    'features__text_pipeline__vect__max_df': (0.5, 1.0),
    #    'features__text_pipeline__vect__max_features': (None, 5000, 10000),
    #    'features__text_pipeline__tfidf__use_idf': (True, False),
    #    'clf__estimator__n_estimators': [50, 100, 200],
    #    'clf__estimator__min_samples_split': [2, 4]
    #}
	#cv = GridSearchCV(pipeline, param_grid=parameters)
	return pipeline


def evaluate_model(model, category_names, X_test, Y_test):
	'''
	INPUT:
	model - model to evaluate
	category_names - all possible classification names as list
	X_test, Y_test - test dataset
	category_names - all possible classification 
	'''
	y_test_pred = model.predict(X_test)
	for i in range(len(Y_test.T)):
		print(category_names[i]+": "+classification_report(Y_test.iloc[:,i], y_test_pred.T[i])+'\n') 

# save model as .pkl file to model_filepath
def save_model(model, model_filepath):
	'''
	INPUT: 
	model - save this model
	model_filepath - to this filepath
	'''
	pickle.dump(model, open(model_filepath, 'wb'))


def main():
	'''
	Execute training classifier and save model to model_filepath
	'''
	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]
		print('Loading data...\n    DATABASE: {}'.format(database_filepath))
		X, Y, category_names = load_data(database_filepath)
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
		print('Building model...')
		model = build_model()
        
		print('Training model...')
		model.fit(X_train, Y_train)
		print('Saving model...\n    MODEL: {}'.format(model_filepath))
		save_model(model, model_filepath)
		print('Evaluating model...')
		evaluate_model(model, category_names, X_test, Y_test)
		print('Trained model saved!')
	else:
		print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()