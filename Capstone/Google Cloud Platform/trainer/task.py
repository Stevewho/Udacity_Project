from trainer.helper import *
from trainer.loader import load_data
import numpy as np
import pandas as pd
from scipy import sparse

#for NN model
from keras.layers.advanced_activations import PReLU
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from time import time
import datetime
from sklearn.model_selection import StratifiedKFold
import os.path
import argparse
import datalab.storage as gcs


	
def main(data_dir):
	train, test, train_label, test_id= load_data(data_dir)

	#validation fold
	NFOLDS = 5
	kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)


	#find missing value by each row and recode to column 'missing'
	train['missing'] = (train==-1).sum(axis=1).astype(float)
	test['missing'] = (test==-1).sum(axis=1).astype(float)
	interval_fea, ordinal_fea, binary_fea, nominal_fea = features_type(train)

	#feature importance
	impo_fea=feature_import(train, train_label)


	#features assemble
	train_nominal, test_nominal = Nominal_Encode(nominal_fea, train, test )
	train_ordinal = train[ordinal_fea]
	test_ordinal = test[ordinal_fea]
	train_bin = train[binary_fea]
	test_bin =test[binary_fea]
	train_intv= train[interval_fea]
	test_intv=test[interval_fea]
	train_miss = train['missing'].values.reshape(-1,1)
	test_miss=test['missing'].values.reshape(-1,1)

	#feature encoding 
	train_tef, test_tef = target_pro(train, test,train_label, interval_fea, nominal_fea)

	#aggregate featurs
	train_list = [train_nominal, train_ordinal, train_bin, train_intv, train_miss, train_tef] 
	test_list = [test_nominal , test_ordinal, test_bin, test_intv, test_miss, test_tef]
	X = sparse.hstack(train_list).tocsr().toarray()

	#test data procssing
	tem_test = sparse.hstack(test_list).tocsr().toarray()
	X_test=[]
	train_impo= train[impo_fea]
	test_impo= test[impo_fea]
	for i in range(train_impo.shape[1]):
	    X_test.append(test_impo.values[:,i].reshape(-1,1))
	X_test.append(tem_test)

	def nn_model():
	    inputs = []
	    layers = []
	    for _ in impo_fea:
	        input_impo = Input(shape=(1, ), dtype='float32')
	        net_impo = Dense(64, kernel_initializer='glorot_normal')(input_impo)
	        net_impo = PReLU()(net_impo)
	        net_impo = BatchNormalization()(net_impo)
	        net_impo = Dense(32, kernel_initializer='glorot_normal')(input_impo)
	        net_impo = PReLU()(net_impo)
	        net_impo = BatchNormalization()(net_impo)
	        net_impo = Dropout(0.25)(net_impo)
	        inputs.append(input_impo)
	        layers.append(net_impo)

	    input_num = Input(shape=(X.shape[1],), dtype='float32')
	    layers.append(input_num)
	    inputs.append(input_num)

	    flatten = Concatenate(axis=-1)(layers)

	    fc1 = Dense(512, kernel_initializer='glorot_normal')(flatten)
	    fc1 = PReLU()(fc1)
	    fc1 = BatchNormalization()(fc1)
	    fc1 = Dropout(0.5)(fc1)
	    fc1 = Dense(64, kernel_initializer='glorot_normal')(fc1)
	    fc1 = PReLU()(fc1)
	    fc1 = BatchNormalization()(fc1)
	    fc1 = Dropout(0.25)(fc1)

	    outputs = Dense(1, kernel_initializer='glorot_normal', activation='sigmoid')(fc1)
	    model = Model(inputs = inputs, outputs = outputs)
	    model.compile(loss='binary_crossentropy', optimizer='adam')
	    return model

	#start train
	cv_train = np.zeros(len(train_label))
	cv_pred = np.zeros(len(test_id))

	for (train_index, valid_index) in kfold.split(X, train_label):
	    x_train = X[train_index]
	    y_train = train_label[train_index]
	    x_valid = X[valid_index]
	    y_valid = train_label[valid_index]

	    x_train_impo = train_impo.values[train_index]
	    x_valid_impo = train_impo.values[valid_index]


	   
	    train_list, valid_list = [], []
	    for i in range(x_train_impo.shape[1]):
	        train_list.append(x_train_impo[:, i].reshape(-1, 1))
	        valid_list.append(x_valid_impo[:, i].reshape(-1, 1))

	    train_list.append(x_train)
	    valid_list.append(x_valid)
	    model = nn_model()

	    model.fit(train_list, y_train, epochs=20, batch_size=512, verbose=2, validation_data=[valid_list, y_valid])
	    cv_train[valid_index] += model.predict(x=valid_list, batch_size=512, verbose=0)[:,0]
	    print('Current Folder Scores: ',Gini(train_label[valid_index], cv_train[valid_index]))
	    cv_pred += model.predict(x=X_test, batch_size=512, verbose=0)[:,0]

	print(Gini(train_label, cv_train ))
	result = pd.DataFrame({'id': test_id, 'target': cv_pred * 1./ (NFOLDS)})
	return result



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data-dir',
		help = 'GCS or local path to training and testing data',
		required =True
		)
	parser.add_argument(
		'--output-name',
		help = 'GCS or local path to training and testing data',
		required =True
		)
	args = parser.parse_args()
	
	# v1
	# arguments = args.__dict__
	# data_dir = arguments.pop("data_dir")
	# output_name= arguments.pop("output_name")
	#v2
	data_dir=args.data_dir
	output_name=args.output_name
	
	outputpath= "Potro/output/"+str(output_name)+'.csv'
	result = main(data_dir)
	gcs.Bucket('stevenwho').item(outputpath).write_to(result.to_csv(),'text/csv')


	






