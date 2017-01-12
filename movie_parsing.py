import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def parse(df_movies):
	""" Pre-process the movie's attribute to vector of selected attributes
	Args: 
		df_movies (dataframe) : dataframe of movie attributes to be parsed
	Returns: 
		df_movies (dataframe) : parsed dataframe of movie attributes
	"""
	def num_list_parse(x):
		try:
			return ast.literal_eval(x)
#         return [n.strip() for n in x]
		except:
			return np.nan
	def str_list_parse(x):
		try:
			parsed_list = [n.strip() for n in x.rstrip(']').lstrip('[').split(',')]
			return ",".join(parsed_list)
		except:
			return "Null"
	def entity_parse(x):
		try:
			return ",".join(re.findall(r"\((\d+),",x))
		except:
			return "Null"


	df_movies['genres'] = df_movies['genres'].apply(str_list_parse)
	df_movies['plot'].fillna("",inplace=True)
	#df_movies['languages'] = df_movies['languages'].apply(str_list_parse)
	df_movies['cast'] = df_movies['cast'].apply(entity_parse)
	df_movies['director'] = df_movies['director'].apply(entity_parse)
	df_movies['production_company'] = df_movies['production_company'].apply(entity_parse)


	return df_movies[['imdbId','plot','year','votes','genres','cast','director','production_company']]

	
