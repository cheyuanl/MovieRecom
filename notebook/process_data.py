import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csc_matrix

def process(df_movies,df_users,movie_features='all',thresholds=(0.1,0.8)):
	""" Pre-process the movie's attribute to vector of selected attributes
		Args: 
			df_movies (dataframe) : dataframe of movie attributes
			df_users (dataframe) : dataframe of users attributes
			movie_features (list): features of movies to be returned
			thresholds: tuple of thresholds for TFIDFs of plot feature
		Returns: 
			df_movies (dataframe) : dataframe of movie attributes
			df_users  (dataframe) : dataframe of user attributes
	"""
	
	#process df_users
	
	#one hot encoding
	df_users['gender'] = df_users['gender'].apply(lambda x: csc_matrix(np.array([1])) if x == 'F' else csc_matrix(np.array([0])))
	df_users['userId_onehot'] = [csc_matrix(i) for i in pd.get_dummies(df_users['userId'].values).values]
	df_users['occupation_onehot'] = [csc_matrix(i) for i in pd.get_dummies(df_users['occupation'].values).values]

	##process df_movies
	if movie_features == 'all':
		movie_features = ['genres','plot','votes','year','cast','director','production_company', 'imdbId_onehot']
			
	if 'imdbId_onehot' in movie_features:
		df_movies['imdbId_onehot'] = [csc_matrix(i) for i in pd.get_dummies(df_movies.loc[:, 'imdbId'].values).values]

	if 'genres' in movie_features:
		df_movies['genres_onehot'] =  [csc_matrix(i) for i in df_movies.loc[:, 'genres'].str.get_dummies(sep=',').values]   

	if 'plot' in movie_features:	
		tfidf = TfidfVectorizer(min_df=thresholds[0], max_df=thresholds[1], ngram_range=(1,3),stop_words='english')
		X_tfidf = tfidf.fit_transform(df_movies.loc[:, 'plot'].values)
		df_movies["plot_tfidf"] = [i for i in X_tfidf]

	if 'votes' in movie_features:
		df_movies['votes'] = [csc_matrix(np.array([i])) for i in df_movies.loc[:, 'votes'].values]
	
	if 'year' in movie_features:
		# df['year'] = df_movies['year'].apply(lambda x: csc_matrix(np.array([x])))
		df_movies['year'] = [csc_matrix(np.array([i])) for i in df_movies.loc[:, 'year'].values]

	if 'cast' in movie_features:
		tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,3),stop_words='english')
		X_tfidf = tfidf.fit_transform(df_movies.loc[:, 'cast'].values)
		df_movies["cast_tfidf"] = [i for i in X_tfidf]

	if 'director' in movie_features:
		tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,3),stop_words='english')
		X_tfidf = tfidf.fit_transform(df_movies.loc[:, 'director'].values)
		df_movies["director_tfidf"] = [i for i in X_tfidf]

	if 'production_company' in movie_features:
		tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,3),stop_words='english')
		X_tfidf = tfidf.fit_transform(df_movies.loc[:, 'production_company'].values)
		df_movies["production_company"] = [i for i in X_tfidf]


	return df_movies, df_users.drop(['zip_code'], 1)

	
	