from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

tex=fetch_20newsgroups(remove=("headers","footers","quotes"))

tf_vec=TfidfVectorizer()
X_train_counts=tf_vec.fit_transform(tex.data)
model=LatentDirichletAllocation(n_components=20)
model.fit(X_train_counts)


count_vec=CountVectorizer()
X_train_counts=count_vec.fit_transform(tex.data)
model2=LatentDirichletAllocation(n_components=20)
model2.fit(X_train_counts)


feature_names = tf_vec.get_feature_names()
sorting = np.argsort(model.components_, axis=1)[:, ::-1]
mglearn.tools.print_topics(topics=range(20),
                           feature_names=np.array(feature_names),
                           topics_per_chunk=20,
                           sorting=sorting,n_words=10)
