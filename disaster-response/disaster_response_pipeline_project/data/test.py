import sys
import pandas as pd
from sqlalchemy import create_engine
# load data from database
def load_data():
    '''
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    conn = engine.connect()
    df = pd.read_sql('SELECT * FROM DisasterResponse', con = conn)
    X = df['message']
    y = df.drop(['id','message','original'], axis=1)
    return X, y


def main():
	X, y = load_data()
	print(X.shape)
	print(y.shape)

if __name__ == '__main__':
    main()