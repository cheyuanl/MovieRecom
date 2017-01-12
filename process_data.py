import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
	
	#df_users.drop(['index'], axis=1, inplace=True)
	#one hot encoding
	df_users['gender'] = df_users['gender'].apply(lambda x: 1 if x == 'F' else 0)
	#occupation dummies
	occupation_dummies = pd.get_dummies(df_users['occupation'])
	df_users = pd.concat([df_users, occupation_dummies], axis=1)
	
	##process df_movies
	
	df = df_movies['imdbId']
	
	if movie_features == 'all':
		movie_features = ['genres','plot','votes','year','cast','director','production_company']
			
	if 'genres' in movie_features:
		#one hot encoding for genres
		genres_dummies = df_movies['genres'].str.get_dummies(sep=',')
		df = pd.concat([df, genres_dummies], axis=1)
	if 'plot' in movie_features:	
		tfidf = TfidfVectorizer(min_df=thresholds[0], max_df=thresholds[1], ngram_range=(1,3),stop_words='english')
	
		X_tfidf = tfidf.fit_transform(df_movies['plot'].values)
		df["plot_tfidf"] = [i.toarray() for i in X_tfidf]
	if 'votes' in movie_features:
		df['votes'] = df_movies['votes']
	if 'year' in movie_features:
		df['year'] = df_movies['year']
	if 'cast' in movie_features:
		tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,3),stop_words='english')
		X_tfidf = tfidf.fit_transform(df_movies['cast'].values)
		df["cast_tfidf"] = [i.toarray() for i in X_tfidf]			
	if 'director' in movie_features:
		tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,3),stop_words='english')
		X_tfidf = tfidf.fit_transform(df_movies['director'].values)
		df["director_tfidf"] = [i.toarray() for i in X_tfidf]
	if 'production_company' in movie_features:
		tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,3),stop_words='english')
		X_tfidf = tfidf.fit_transform(df_movies['production_company'].values)
		df["production_company"] = [i.toarray() for i in X_tfidf]

	return df,df_users.drop('occupation',1)

	
	