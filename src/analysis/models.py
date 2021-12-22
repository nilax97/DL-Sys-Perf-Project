from analysis.utils import get_time_train_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import os
import pickle

def get_trained_model(model_type,folder_name):
	x,y = get_time_train_data(model_type,folder_name)
	min_error = float('inf')
	min_model = None
	for i in range(100):
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
		rf = RandomForestRegressor(n_estimators=20)
		rf.fit(x_train, y_train)
		y_pred_train = rf.predict(x_train)
		y_pred_test = rf.predict(x_test)
		train_error = mean_absolute_percentage_error(y_train,y_pred_train)
		test_error = mean_absolute_percentage_error(y_test,y_pred_test)
		if test_error < min_error:
			min_loss = test_error
			min_model = rf
		if (test_error - train_error) < 0.05 and max(test_error,train_error) < 0.1:
			break
	y_pred_train = min_model.predict(x_train)
	y_pred_test = min_model.predict(x_test)
	print(f'{model_type} - train MSE : {mean_absolute_percentage_error(y_train,y_pred_train)}')
	print(f'{model_type} - test MSE : {mean_absolute_percentage_error(y_test,y_pred_test)}')
	
	return min_model

def get_models():
	if os.path.exists("results/trained_model.pickle"):

		with open("results/trained_model.pickle", 'rb') as handle:
			models = pickle.load(handle)
	else:
		models = dict()
		models['vgg'] = get_trained_model('vgg',"pickle")
		models['inception'] = get_trained_model('inception',"pickle")
		models['resnet'] = get_trained_model('resnet',"pickle")
		models['fc'] = get_trained_model('fc',"pickle")
		with open("results/trained_model.pickle", 'wb') as handle:
			pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return models