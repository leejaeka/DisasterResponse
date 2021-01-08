import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	messages = pd.read_csv('data/messages.csv')
    categories = pd.read_csv('data/categories.csv')


def clean_data(df):

	# Convert category values and columns names appropriately
    df = messages.merge(categories, how='left', on=['id'])
	categories = df['categories'].str.split(';',expand=True)
	row = categories.iloc[0]
	category_colnames = row.apply(lambda x: x.split('-')[0])
	categories.columns = category_colnames
	for column in categories:
		categories[column] = categories[column].apply(lambda x : x.split('-')[1])
		categories[column] = categories[column].astype(str)
	df = pd.concat([df.drop(['categories'], axis=1), categories], axis=1)
	df = df.drop_duplicates()
def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse.db')
	df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()