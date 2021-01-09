import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	'''
	INPUT:
	messages_filepath - filepath of the messages column
	categories_filepath - filepath of the categories columns
	OUTPUT:
	df - merged dataframe of messages and categories
	'''
	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath)
	df = messages.merge(categories, how='left', on=['id'])
	return df

def clean_data(df):
	'''
	INPUT:
	df - dataframe to clean
	OUTPUT:
	df - cleaned data 
	'''
	# Convert category values and columns names appropriately
	categories = df['categories'].str.split(';',expand=True)
	row = categories.iloc[0]
	category_colnames = row.apply(lambda x: x.split('-')[0])
	categories.columns = category_colnames
	for column in categories:
		categories[column] = categories[column].apply(lambda x : x.split('-')[1])
		categories[column] = categories[column].astype(int)
	df = pd.concat([df.drop(['categories'], axis=1), categories], axis=1)
	df = df[df['related'] != 2] # remove all rows with non binary value
	df = df.drop_duplicates() # drop all duplicates
	return df

def save_data(df, database_filename):
	'''
	INPUT:
	df - dataframe to save
	database_filename - name to be saved to sqlite db 
	'''
	engine = create_engine('sqlite:///' + database_filename)
	df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
	'''
	Executes cleaning and saving data to sql database
	'''
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