# Environment ------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
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
import itertools

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


# Hierachical Clustering -------------------------------------------------------
# loop over combinations of distance and linkage functions
def plot_dendrogram(
    df, output_name, y_intercept=None, metric='euclidean', linkage='ward'
):
    plt.rcParams['font.family'] = 'Source Han Sans HC'
    dist = sch.linkage(df, metric=metric, method=linkage)
    plt.figure(figsize=(10, 7))
    sch.dendrogram(
        dist,
        orientation='top',
        labels=df.index,
        distance_sort='descending',
        show_leaf_counts=True
    )
    if y_intercept is not None:
        plt.axhline(y=y_intercept, c='grey', lw=1, linestyle='dashed')
    plt.savefig(
        Path('outputs', 'figs', f'{output_name}_{metric}_{linkage}.png'),
        dpi=300,
        bbox_inches='tight'
    )

    silhouette_coefficients = []
    for k in range(2, 11):
        cluster = AgglomerativeClustering(
            n_clusters=k,
            affinity=metric,
            linkage=linkage
        )
        cluster.fit(df)
        score = silhouette_score(df, cluster.labels_)
        silhouette_coefficients.append(score)
        plt.figure(figsize=(10, 7))

    plt.bar(range(2, 11), silhouette_coefficients)
    plt.xlabel('Number of Clusters', fontsize=20)
    plt.ylabel('Silhoette Coefficients', fontsize=20)
    plt.savefig(
        Path('outputs', 'figs',
             f'{output_name}_{metric}_{linkage}_silhoette.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close('all')


metrics = ['euclidean', 'cityblock']
methods = ['single', 'complete', 'median',
           'weighted', 'average', 'centroid', 'ward']

for mm in list(itertools.product(metrics, methods)):
    try:
        plot_dendrogram(
            emotional_words_pd,
            output_name='cue-emotion_hclust',
            metric=mm[0],
            linkage=mm[1]
        )
    except:
        pass
# all algorithms gave optimal solutions: positive and negative


# Affective Features 2 Clusters ------------------------------------------------
# plot dendrogram
plt.rcParams['font.family'] = 'Source Han Sans HC'
dist = sch.linkage(emotional_words_pd, metric='euclidean', method='ward')
plt.figure(figsize=(10, 7))
sch.dendrogram(dist,
               orientation='top',
               labels=emotional_words_pd.index,
               distance_sort='descending',
               show_leaf_counts=True)
plt.axhline(y=15, c='grey', lw=1, linestyle='dashed')
plt.savefig(Path('outputs', 'figs', 'cue-emotion_hclust_ward_2clusters.png'),
            dpi=300, bbox_inches='tight')

# clustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
                                  linkage='ward')
cluster.fit_predict(emotional_words_pd)

cluster1_list = [emotional_words_pd.index[i]
                 for i in range(len(cluster.labels_)) if cluster.labels_[i] == 0]
cluster2_list = [emotional_words_pd.index[i]
                 for i in range(len(cluster.labels_)) if cluster.labels_[i] == 1]

cluster1_list
cluster2_list

cols = emotional_words_pd.columns
# add memberships to the emotion dataframe
emotional_words_pd_copy = emotional_words_pd.copy()
emotional_words_pd_copy['group'] = cluster.labels_

# make sure all variable are numeric
emotional_words_pd_copy[cols] = emotional_words_pd_copy[cols].apply(
    pd.to_numeric, errors='coerce')

cluster1_rating_avg = (emotional_words_pd_copy
                       .query('cues == @cluster1_list')
                       .mean(axis=0))
cluster1_rating_sd = (emotional_words_pd_copy
                      .query('cues == @cluster1_list')
                      .std(axis=0))

cluster2_rating_avg = (emotional_words_pd_copy
                       .query('cues == @cluster2_list')
                       .mean(axis=0))
cluster2_rating_sd = (emotional_words_pd_copy
                      .query('cues == @cluster2_list')
                      .std(axis=0))

# compare 2 clusters
for col_name in cols:

    print(col_name)
    model = ols(f'{col_name} ~ C(group)', emotional_words_pd_copy).fit()
    es = anova_lm(model, typ=1)
    print(utils.anova_table(es))
    comparison = MultiComparison(
        emotional_words_pd_copy[f'{col_name}'], emotional_words_pd_copy['group'])
    comparison_results = comparison.tukeyhsd()
    print(comparison_results.summary())

hclust_2_long = pd.melt(
    emotional_words_pd_copy,
    id_vars='group',
    value_vars=['valence_mean', 'arousal_mean',
                'dominance_mean', 'concreteness_mean']
).replace(
    dict(
        variable={
            'valence_mean': 'Valence',
            'arousal_mean': 'Arousal',
            'dominance_mean': 'Dominance',
            'concreteness_mean': 'Concreteness'
        },
        group={
            0: 'Cluster B',
            1: 'Cluster A'
        }
    )
)
hclust_2_long['group'] = hclust_2_long['group'].astype(
    "category").cat.set_categories(['Cluster A', 'Cluster B'], ordered=True)

plt.figure(figsize=(6, 3))
fig = sns.barplot(data=hclust_2_long, x='variable', y='value', hue='group')
plt.xlabel('')
plt.ylabel('Value')
fig.legend_.set_title(None)
plt.savefig(Path('outputs', 'figs', 'emotion_2clusters.png'),
            dpi=300, bbox_inches='tight')

# Affective Features 3 Clusters ------------------------------------------------
# plot dendrogram
plt.rcParams['font.family'] = 'Source Han Sans HC'
dist = sch.linkage(emotional_words_pd, metric='euclidean', method='ward')
plt.figure(figsize=(10, 7))
sch.dendrogram(dist,
               orientation='top',
               labels=emotional_words_pd.index,
               distance_sort='descending',
               show_leaf_counts=True)
plt.axhline(y=5, c='grey', lw=1, linestyle='dashed')
plt.savefig(Path('outputs', 'figs', 'cue-emotion_hclust_ward_2clusters.png'),
            dpi=300, bbox_inches='tight')

# clustering
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

cols = emotional_words_pd.columns
# add memberships to the emotion dataframe
emotional_words_pd_copy = emotional_words_pd.copy()
emotional_words_pd_copy['group'] = cluster.labels_


# make sure all variable are numeric
emotional_words_pd_copy[cols] = emotional_words_pd_copy[cols].apply(
    pd.to_numeric, errors='coerce')

cluster1_rating_avg = (emotional_words_pd_copy
                       .query('cues == @cluster1_list')
                       .mean(axis=0))
cluster1_rating_sd = (emotional_words_pd_copy
                      .query('cues == @cluster1_list')
                      .std(axis=0))

cluster2_rating_avg = (emotional_words_pd_copy
                       .query('cues == @cluster2_list')
                       .mean(axis=0))
cluster2_rating_sd = (emotional_words_pd_copy
                      .query('cues == @cluster2_list')
                      .std(axis=0))

cluster3_rating_avg = (emotional_words_pd_copy
                       .query('cues == @cluster3_list')
                       .mean(axis=0))
cluster3_rating_sd = (emotional_words_pd_copy
                      .query('cues == @cluster3_list')
                      .std(axis=0))

# compare 3 clusters
for col_name in cols:

    print(col_name)
    model = ols(f'{col_name} ~ C(group)', emotional_words_pd_copy).fit()
    es = anova_lm(model, typ=1)
    print(utils.anova_table(es))
    comparison = MultiComparison(
        emotional_words_pd_copy[f'{col_name}'], emotional_words_pd_copy['group'])
    comparison_results = comparison.tukeyhsd()
    print(comparison_results.summary())

# plot 3 cluster solutins
hclust_3_long = pd.melt(
    emotional_words_pd_copy,
    id_vars='group',
    value_vars=['valence_mean', 'arousal_mean',
                'dominance_mean', 'concreteness_mean']
).replace(
    dict(
        variable={
            'valence_mean': 'Valence',
            'arousal_mean': 'Arousal',
            'dominance_mean': 'Dominance',
            'concreteness_mean': 'Concreteness'
        },
        group={
            0: 'Cluster A',
            1: 'Cluster B',
            2: 'Cluster C'
        }
    )
)
hclust_3_long['group'] = hclust_3_long['group'].astype(
    "category").cat.set_categories(['Cluster A', 'Cluster B', 'Cluster C'], ordered=True)

plt.figure(figsize=(6, 3))
fig = sns.barplot(data=hclust_3_long, x='variable', y='value', hue='group')
plt.xlabel('')
plt.ylabel('Value')
fig.legend_.set_title(None)
plt.savefig(Path('outputs', 'figs', 'emotion_3clusters.png'),
            dpi=300, bbox_inches='tight')


# Standardization affect results? ----------------------------------------------
scaler = StandardScaler()
emotional_words_pd_std = scaler.fit_transform(emotional_words_pd.copy())
emotional_words_pd_std = pd.DataFrame(
    emotional_words_pd_std,
    columns=['valence_mean', 'arousal_mean',
             'dominance_mean', 'concreteness_mean']
)

for mm in list(itertools.product(metrics, methods)):
    try:
        plot_dendrogram(
            emotional_words_pd_std,
            output_name='cue-emotion_hclust_std',
            metric=mm[0],
            linkage=mm[1]
        )
    except:
        pass


plt.rcParams['font.family'] = 'Source Han Sans HC'
dist = sch.linkage(emotional_words_pd_std, metric='euclidean', method='ward')
plt.figure(figsize=(10, 7))
sch.dendrogram(dist,
               orientation='top',
               labels=emotional_words_pd.index,
               distance_sort='descending',
               show_leaf_counts=True)
plt.savefig(Path('outputs', 'figs', 'cue-emotion_hclust_std_ward.png'),
            dpi=300, bbox_inches='tight')

# clustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean',
                                  linkage='ward')
cluster.fit_predict(emotional_words_pd_std)

cluster1_list = [emotional_words_pd.index[i]
                 for i in range(len(cluster.labels_)) if cluster.labels_[i] == 0]
cluster2_list = [emotional_words_pd.index[i]
                 for i in range(len(cluster.labels_)) if cluster.labels_[i] == 1]
cluster3_list = [emotional_words_pd.index[i]
                 for i in range(len(cluster.labels_)) if cluster.labels_[i] == 2]

cluster1_list
cluster2_list
cluster3_list

cols = emotional_words_pd.columns
# add memberships to the emotion dataframe
emotional_words_pd_copy = pd.DataFrame(
    emotional_words_pd_std,
    columns=['valence_mean', 'arousal_mean',
             'dominance_mean', 'concreteness_mean']
)
emotional_words_pd_copy['group'] = cluster.labels_

# make sure all variable are numeric
emotional_words_pd_copy[cols] = emotional_words_pd_copy[cols].apply(
    pd.to_numeric, errors='coerce')

cluster1_rating_avg = (emotional_words_pd_copy
                       .query('cues == @cluster1_list')
                       .mean(axis=0))
cluster1_rating_sd = (emotional_words_pd_copy
                      .query('cues == @cluster1_list')
                      .std(axis=0))

cluster2_rating_avg = (emotional_words_pd_copy
                       .query('cues == @cluster2_list')
                       .mean(axis=0))
cluster2_rating_sd = (emotional_words_pd_copy
                      .query('cues == @cluster2_list')
                      .std(axis=0))

cluster3_rating_avg = (emotional_words_pd_copy
                       .query('cues == @cluster3_list')
                       .mean(axis=0))
cluster3_rating_sd = (emotional_words_pd_copy
                      .query('cues == @cluster3_list')
                      .std(axis=0))

# compare 3 clusters
for col_name in cols:

    print(col_name)
    model = ols(f'{col_name} ~ C(group)', emotional_words_pd_copy).fit()
    es = anova_lm(model, typ=1)
    print(utils.anova_table(es))
    comparison = MultiComparison(
        emotional_words_pd_copy[f'{col_name}'], emotional_words_pd_copy['group'])
    comparison_results = comparison.tukeyhsd()
    print(comparison_results.summary())
