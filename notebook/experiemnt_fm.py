from pyfm import pylibfm

def iter_error(X_train, y_tr, X_test, y_te, iters = [1,3,5,10,25,50,100,200], lr = 1E-2, num_factor = 3):
    models = []
    train_error, test_error = [], []

    for k in iters:
        model = pylibfm.FM(num_factors=num_factor, num_iter=k,
                        verbose=True, task="regression",
                        initial_learning_rate=lr, learning_rate_schedule="optimal")
        model.fit(X_train, y_tr)
        models.append((k, model))

        pred = model.predict(X_train)
        train_error.append((k, mean_squared_error(y_tr, pred)))

        pred = model.predict(X_test)
        test_error.append((k, mean_squared_error(y_te, pred)))

    return models, train_error, test_error


def laten_error(X_train, y_tr, X_test, y_te, laten = [1,3,5,10,25,50,100,200], lr = 1E-2 , iters = 100):
    models = []
    train_error, test_error = [], []

    for k in laten:
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