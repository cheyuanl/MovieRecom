from pyfm import pylibfm
from sklearn.metrics import mean_squared_error

def iter_error(X_train, y_tr, X_test, y_te, iters = [1,3,5,10,25,50,100,200], lr = 1E-2, latent = 3):
	""" Test the MSE error against number of iteration.
		Args:
			X_train (2d sparse matrix) : training data
			y_tr (numpy array) training : target
			X_test (2d sparse matrix) : testing data
			y_te (numpy arry) : testing target
			iters (list) : number of iterations
			lr (float) : learning rate
			num_factors : number of laten variable
		Result:
			models (fm model) : factorization machine model
			train_error (list of tuple) : [(k, error), ...]
			test_error (list of tuple) : [(k, error), ...]
	"""
	models = []
	train_error, test_error = [], []

	for k in iters:
		model = pylibfm.FM(num_factors=latent, num_iter=k,
							verbose=True, task="regression",
							initial_learning_rate=lr, learning_rate_schedule="optimal")
		model.fit(X_train, y_tr)
		models.append((k, model))

		pred = model.predict(X_train)
		train_error.append((k, mean_squared_error(y_tr, pred)))

		pred = model.predict(X_test)
		test_error.append((k, mean_squared_error(y_te, pred)))

	return models, train_error, test_error


def latent_error(X_train, y_tr, X_test, y_te, latent = [1,3,5,10,25,50,100,200], lr = 1E-2 , iters = 100):
	""" Test the MSE error against number of latent factor.
	Args:
		X_train (2d sparse matrix) : training data
		y_tr (numpy array) training : target
		X_test (2d sparse matrix) : testing data
		y_te (numpy arry) : testing target
		iters (list) : number of iterations
		lr (float) : learning rate
		num_factors : number of laten variable
	Result:
		models (fm model) : factorization machine model
		train_error (list of tuple) : [(k, error), ...]
		test_error (list of tuple) : [(k, error), ...]
	"""
	models = []
	train_error, test_error = [], []

	for k in latent:
		model = pylibfm.FM(num_factors=k, num_iter=iters,
		                verbose=True, task="regression",
		                initial_learning_rate=lr, learning_rate_schedule="optimal")
		model.fit(X_train, y_tr)
		models.append((k, model))

		pred = model.predict(X_train)
		train_error.append((k, mean_squared_error(y_tr, pred)))

		pred = model.predict(X_test)
		test_error.append((k, mean_squared_error(y_te, pred)))

	return models, train_error, test_error