#!/usr/bin/env python
# coding: utf-8

# In[18]:


import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


# In[19]:


url = 'https://drive.google.com/file/d/19E-Sh2TomdKqIpyWXLM3FZMQnheP0dCm/view?usp=sharing'
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
movies = pd.read_csv(url)
movies.head(10)


# In[20]:


selected_features = ['overview', 'genre']
print(selected_features)


# In[21]:


for feature in selected_features:
  movies[feature] = movies[feature].fillna('')


# In[22]:


movies=movies[['id', 'title', 'overview', 'genre']]


# In[23]:


movies['tags'] = movies['overview']+movies['genre']
movies


# In[24]:


combined_features= movies['tags']


# In[25]:


new_data  = movies.drop(columns=['overview', 'genre'])
new_data


# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)


# In[27]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(feature_vectors)
print(similarity)


# In[28]:


print(similarity.shape)


# In[29]:


movie_name = input(' Enter your favourite movie name : ')


# In[30]:


import difflib
if 'index' not in movies.columns:
    movies.reset_index(inplace=True)
list_of_all_titles = movies['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]
index_of_the_movie = movies[movies.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)
print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies[movies.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[31]:


import pickle


# In[32]:


pickle.dump(new_data, open('movies_list.pkl', 'wb'))


# In[33]:


pickle.dump(similarity, open('similarity.pkl', 'wb'))


# In[34]:


pickle.load(open('movies_list.pkl', 'rb'))

