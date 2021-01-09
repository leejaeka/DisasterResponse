import json
import plotly
import pandas as pd
import plotly.express as px

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
	'''
	tokenize text into lowercased and stripped and lemmatized list
	INPUT:
	text - text to tokenize
	OUTPUT:
	clean_tokens - tokenized list
	'''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
conn = engine.connect()
df = pd.read_sql('SELECT * FROM DisasterResponse', con = conn)
y = df.drop(['id','message','original','genre'], axis=1)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
	'''
	Create Plotly graphs and render the app with html file
	'''
	# extract data needed for visuals
	count_list = []
	for col in y:
		count_list.append(sum(y[col]))
	# create visuals
	layout_one = dict(title = 'Category Count Bar Plot')
	graphs = [
		dict(data=px.bar(x=y.columns, y=count_list, title="Category Count Bar Plot"), layout=layout_one)
    ]
    
    # encode plotly graphs in JSON
	ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
	graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
	return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
	'''
	call functions to calculate classification and render respective html file
	'''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #app.run(host='0.0.0.0', port=3001, debug=True)
	app.run(host='127.0.0.1', port=3001, debug=True)

if __name__ == '__main__':
    main()