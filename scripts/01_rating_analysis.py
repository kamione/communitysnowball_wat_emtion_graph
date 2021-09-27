# Environment ------------------------------------------------------------------
from sklearn.cluster import AgglomerativeClustering
from networkx.algorithms import similarity
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import distance
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from matplotlib import font_manager
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from src.python.utils import utils

font_files = font_manager.findSystemFonts(fontpaths=str(Path('fonts')))
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

matplotlib.rc("font", family='Source Han Sans HC')
sns_c = sns.color_palette(palette='deep')
sns.set_style('whitegrid', {'axes.grid': False})

emotion_list = pd.read_csv(
    Path('data', 'raw', 'cue_list_updated.csv'), encoding='big5')
emotion_stacked_list = emotion_list.stack(dropna=True).to_list()
#anger_list = emotion_list['Anger'].dropna().to_list()
#digust_list = emotion_list['Disgust'].dropna().to_list()
#fear_list = emotion_list['Fear'].dropna().to_list()
#happiness_list = emotion_list['Happiness'].dropna().to_list()
#sadness_list = emotion_list['Sadness'].dropna().to_list()
#surprise_list = emotion_list['Surprise'].dropna().to_list()


# Data I/O ---------------------------------------------------------------------
data_dir = Path('data', 'raw', 'RatingScaleSummary(11Oct2016).xlsx')
index = pd.read_excel(data_dir,
                      dtype='unicode',
                      sheet_name=1,
                      usecols='A',
                      nrows=2420)
index = index.rename({'Unnamed: 0': 'cues'}, axis=1)

sheet_dict = pd.read_excel(data_dir,
                           dtype='unicode',
                           sheet_name=None,
                           usecols='V, W',
                           nrows=2420)

rating = pd.concat(sheet_dict.values(), axis=1, ignore_index=True)

emotional_words_pd = (pd.concat([index, rating], axis=1)
                      .rename(
    {0: 'valence_median',
     1: 'valence_mean',
     2: 'concreteness_median',
     3: 'concreteness_mean',
     4: 'aoa_median',
     5: 'aoa_mean',
     6: 'arousal_median',
     7: 'arousal_mean',
     8: 'dominance_median',
     9: 'dominance_mean'}, axis=1
)
    .query('cues == @emotion_stacked_list')
    .set_index('cues')
    .loc[:, ['valence_mean', 'arousal_mean', 'dominance_mean', 'concreteness_mean']])
# make sure all columns are numeric
emotional_words_pd = emotional_words_pd.astype('float')
emotional_words_pd.to_csv(
    Path('outputs', 'tables', 'emotion_cues_rating.csv'), encoding='big5')

# K-Mean Clustering (Abondoned)
# sse = []
# k_candidates = range(1, 11)
# for k in k_candidates:
#     k_means = KMeans(random_state=1234, n_clusters=k)
#     k_means.fit(emotional_words_pd)
#     sse.append(k_means.inertia_)
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.scatterplot(x=k_candidates, y=sse, s=80, color='grey', ax=ax)
# sns.scatterplot(x=[k_candidates[2]], y=[sse[2]], color=sns_c[3], s=150, ax=ax)
# sns.lineplot(x=k_candidates, y=sse, alpha=0.5, color='grey', ax=ax)
# ax.set(title='', ylabel='Sum of Squared Error', xlabel='Number of Clusters')
# fig.savefig(Path('outputs', 'figs', 'kmeans_screeplot.png'))
# k_means = KMeans(random_state=1111, n_clusters=3)
# k_means.fit(emotional_words_pd)
# cluster = k_means.predict(emotional_words_pd)

# cluster_list = cluster.tolist()
# cluster1_list = [emotional_words_pd.index[i]
#                  for i in range(len(cluster_list)) if cluster_list[i] == 0]
# cluster2_list = [emotional_words_pd.index[i]
#                  for i in range(len(cluster_list)) if cluster_list[i] == 1]
# cluster3_list = [emotional_words_pd.index[i]
#                  for i in range(len(cluster_list)) if cluster_list[i] == 2]

# Hierachical Clustering -------------------------------------------------------
plt.rcParams['font.family'] = 'Source Han Sans HC'
dist = sch.linkage(emotional_words_pd, metric='euclidean', method='ward')
plt.figure(figsize=(10, 7))
sch.dendrogram(dist,
               orientation='top',
               labels=emotional_words_pd.index,
               distance_sort='descending',
               show_leaf_counts=True)
plt.axhline(y=5, c='grey', lw=1, linestyle='dashed')
plt.savefig(Path('outputs', 'figs', 'cue-emotion_hclust.png'),
            dpi=300, bbox_inches='tight')

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean',
                                  linkage='ward')
cluster.fit_predict(emotional_words_pd)

cluster1_list = [emotional_words_pd.index[i]
                 for i in range(len(cluster.labels_)) if cluster.labels_[i] == 0]
cluster2_list = [emotional_words_pd.index[i]
                 for i in range(len(cluster.labels_)) if cluster.labels_[i] == 1]
cluster3_list = [emotional_words_pd.index[i]
                 for i in range(len(cluster.labels_)) if cluster.labels_[i] == 2]

cluster1_list
cluster2_list
cluster3_list

# from sklearn.metrics import silhouette_score
# silhouette_coefficients = []
# for k in range(2, 11):
#    kmeans = KMeans(n_clusters=k)
#    kmeans.fit(emotional_words_pd)
#    score = silhouette_score(emotional_words_pd, kmeans.labels_)
#    silhouette_coefficients.append(score)

# Affective Features -----------------------------------------------------------

cols = emotional_words_pd.columns
# add memberships to the emotion dataframe
emotional_words_pd['group'] = cluster.labels_

# make sure all variable are numeric
emotional_words_pd[cols] = emotional_words_pd[cols].apply(
    pd.to_numeric, errors='coerce')

cluster1_rating_avg = (emotional_words_pd
                       .query('cues == @cluster1_list')
                       .mean(axis=0))
cluster1_rating_sd = (emotional_words_pd
                      .query('cues == @cluster1_list')
                      .std(axis=0))

cluster2_rating_avg = (emotional_words_pd
                       .query('cues == @cluster2_list')
                       .mean(axis=0))
cluster2_rating_sd = (emotional_words_pd
                      .query('cues == @cluster2_list')
                      .std(axis=0))

cluster3_rating_avg = (emotional_words_pd
                       .query('cues == @cluster3_list')
                       .mean(axis=0))
cluster3_rating_sd = (emotional_words_pd
                      .query('cues == @cluster3_list')
                      .std(axis=0))

# compare 3 clusters
for col_name in cols:

    print(col_name)
    model = ols(f'{col_name} ~ C(group)', emotional_words_pd).fit()
    es = anova_lm(model, typ=1)
    print(utils.anova_table(es))
    comparison = MultiComparison(
        emotional_words_pd[f'{col_name}'], emotional_words_pd['group'])
    comparison_results = comparison.tukeyhsd()
    print(comparison_results.summary())
