import pandas as pd
from tensorflow.python.lib.io import file_io

def load_data(data_dir):


	with file_io.FileIO(data_dir +'/train.csv',mode='r') as train_data:
		train = pd.read_csv(train_data)
		train_label = train['target']
		train_id = train['id']
		del train['target'], train['id']

	with file_io.FileIO(data_dir +'/test.csv',mode='r') as test_data:
		test = pd.read_csv(test_data)
		test_id = test['id']
		del test['id']
	return train, test, train_label, test_id

