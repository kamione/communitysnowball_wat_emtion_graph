# Environment ------------------------------------------------------------------
import itertools
import os
import altair as alt
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import random
import researchpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import ttost_ind
from altair_saver import save
from cdlib import algorithms
from pathlib import Path
from datetime import datetime
from collections import Counter
from netgraph import Graph
from matplotlib import font_manager
from src.python.core import wcloud
from src.python.utils import utils


# Add font family: Source Han Sans HC
font_files = font_manager.findSystemFonts(fontpaths=str(Path('fonts')))
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# print out font name
# font_set = {f.name for f in font_manager.fontManager.ttflist}
# for f in font_set:
#    print(f)
# [f for f in fm.fontManager.ttflist if 'Source Han Sans HC' in f.name]

# seaborn plotting environment
sns.set_style('whitegrid')
sns.set(font='Source Han Sans HC')  # for Chinese characters

# Altair Setting
# CAUTION: diable maximum alt maximum row
alt.data_transformers.disable_max_rows()
# alt.renderers.set_embed_options(scaleFactor=10)

# Data I/O ---------------------------------------------------------------------
data_dir = Path('data', 'raw', '20210514_snowball_cleaned_chinese_only.csv')
data = pd.read_csv(data_dir, dtype='unicode')

# only include the original number of cues
selected_years = list(range(2014, 2020))

data_cleaned = (data
                .assign(
                    created_at=data['created_at'].apply(
                        lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')),
                    created_year=data['created_at'].apply(
                        lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M').year),
                    created_month=data['created_at'].apply(
                        lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M').month),
                    age=pd.to_numeric(data['age']),
                    education=pd.to_numeric(data['education'])
                )
                .query('created_year == @selected_years')
                .query('gender != "x"')
                .query('18 <= age <= 60')
                .replace(['fe', 'ma'], ['Female', 'Male'])
                .replace(['不開心', '痛心'], ['唔開心', '心痛'])
                .reset_index(drop=True)
                )

# check if unique number of cues is correct
n_cues = data_cleaned['cue'].nunique()
n_participant = data_cleaned['participant_id'].nunique()  # total partipants
n_responses = data_cleaned['id'].nunique()  # total responses
# data_cleaned.columns
# data_cleaned['keypress_rt3']

# get summary statistics
data_by_partid_gender = data_cleaned.groupby(
    ['participant_id', 'gender']
).size()
# female and male number
data_by_partid_gender.groupby(['gender']).size()
# average response
np.mean(data_by_partid_gender)

data_cleaned['age'].mean()
data_cleaned['age'].std()
data_cleaned.groupby('gender')['age'].mean()
data_cleaned.groupby('gender')['age'].std()

data_cleaned['education'].mean()
data_cleaned['education'].std()
data_cleaned.groupby('gender')['education'].mean()
data_cleaned.groupby('gender')['education'].std()

# Visualization of Demographic Data --------------------------------------------
age_distribution = alt.Chart(data_cleaned).mark_area(
    opacity=0.8,
    interpolate='step'
).encode(
    alt.X(
        'age:Q',
        bin=alt.Bin(maxbins=50),
        axis=alt.Axis(tickCount=10),
        scale=alt.Scale(bins=np.arange(15, 60, 5).tolist()),
        title="Age (Years)"
    ),
    alt.Y('count()', stack=None),
    alt.Color('gender:N',
              scale=alt.Scale(domain=['Female', 'Male'],
                              range=['#F9BF45', '#434343']),
              title="Gender")
).properties(
    width=300,
    height=200
)

edu_distribution = (alt.Chart(data_cleaned)
                    .mark_area(
    opacity=0.8,
    interpolate='step',
    clip=True
)
    .encode(
        alt.X(
            'education:Q',
            bin=alt.Bin(maxbins=100),
            axis=alt.Axis(tickCount=5),
            scale=alt.Scale(domain=(0, 35), bins=np.arange(0, 40, 5).tolist()),
            title="Education Years"
        ),
        alt.Y('count()', stack=None),
        alt.Color('gender:N',
                  scale=alt.Scale(domain=['Female', 'Male'],
                                  range=['#F9BF45', '#434343']),
                  title="Gender")
)
).properties(
    width=300,
    height=200
)

dist_combined_plot = (age_distribution) | (edu_distribution)
dist_combined_plot = (dist_combined_plot
                      .configure_axis(grid=False)
                      .configure_view(strokeWidth=0)
                      .configure_legend(labelLimit=0)
                      )
save(dist_combined_plot, str(Path('outputs', 'figs', 'dist_combined_plot.pdf')))
save(dist_combined_plot, str(Path('outputs', 'figs', 'dist_combined_plot.png')))


# Word Association Network Construction ----------------------------------------
# R123: combine response R1, R2, and R3
r123_wide = data_cleaned.filter(regex='created_year|cue|r1|r2|r3')
r123_long = pd.melt(r123_wide, id_vars=['cue', 'created_year'],
                    value_vars=['r1', 'r2', 'r3'], value_name='R123')
r123_long = r123_long.drop(r123_long[r123_long.R123 == '沒有更多回應'].index)

# create edges dataframe
r123_edges = (r123_long
              .groupby(['cue', 'R123'])
              .agg({'R123': 'size'})
              .rename({'R123': 'Weight'}, axis=1)
              .reset_index()
              )

graph_orig = nx.from_pandas_edgelist(df=r123_edges, source='cue', target='R123',
                                     edge_attr='Weight',
                                     create_using=nx.DiGraph())

g = graph_orig.copy()

n_edges = nx.number_of_edges(g)
n_nodes = nx.number_of_nodes(g)

# print(f'original graph: nodes - {n_nodes}, edges - {n_edges}')

# remove nodes that out degree is 0
selected_nodes = [node for node, outdeg in g.out_degree() if outdeg != 0]
g_cleaned = g.subgraph(selected_nodes)
n_edges = nx.number_of_edges(g_cleaned)
n_nodes = nx.number_of_nodes(g_cleaned)

# print(f'original graph: nodes - {n_nodes}, edges - {n_edges}')

# get the strongly connected component subgraph
strongly_connected_nodes = max(
    nx.strongly_connected_components(g_cleaned), key=len)
g_cleaned_strong = g_cleaned.subgraph(strongly_connected_nodes)
n_edges = nx.number_of_edges(g_cleaned_strong)
n_nodes = nx.number_of_nodes(g_cleaned_strong)

print(f'original graph: nodes - {n_nodes}, edges - {n_edges}')


# Emotion Sub-Network Construction ---------------------------------------------
# load word list
emotion_list = pd.read_csv(
    Path('data', 'raw', 'cue_list_updated.csv'), encoding='big5'
)
emotion_stacked_list = emotion_list.stack(dropna=True).to_list()

n_emotions = len(emotion_stacked_list)
n_emotions

# create edges dataframe only from emotional words
r123_edges_emotions = (
    r123_long
    .groupby(['cue', 'R123'])
    .agg({'R123': 'size'})
    .rename({'R123': 'Weight'}, axis=1)
    .reset_index()
    .query('cue == @emotion_stacked_list')
)

count = r123_edges_emotions["cue"].value_counts()
np.mean(count)

dist_cue_n_responses = alt.Chart(r123_edges_emotions).mark_bar().encode(
    x=alt.X("cue:N", title='Emotional Words', sort='-y'),
    y=alt.Y('count()', title='Number of Responses'),
    color=alt.Y('count()', legend=None),
    opacity=alt.value(0.8)
).configure_axis(
    grid=False
).configure_view(
    strokeWidth=0
).configure_legend(
    labelLimit=0
).properties(
    width=750,
    height=300
)
save(dist_cue_n_responses, str(Path('outputs', 'figs', 'dist_cue_n_responses.pdf')))
save(dist_cue_n_responses, str(Path('outputs', 'figs', 'dist_cue_n_responses.png')))

# get the list of emotion and thier neighbours
select_nodes = list()
for emotion_word in emotion_stacked_list:
    select_nodes.append(emotion_word)
    emotion_neighbor_nodes = [node for node in g.neighbors(emotion_word)]
    select_nodes = select_nodes + emotion_neighbor_nodes

# generate emotion subgraph from main graph
emotion_subgraph = g_cleaned_strong.subgraph(select_nodes)
nx.number_of_nodes(emotion_subgraph)
nx.number_of_edges(emotion_subgraph)

g_emotion_nodes = list(emotion_subgraph.nodes)

# get the hubs of the emotion graph
indegree_list = nx.in_degree_centrality(emotion_subgraph)
indegree_list_top20 = sorted(indegree_list, reverse=True)[:21]
pangerank_list = nx.pagerank(emotion_subgraph, alpha=0.8)
pangerank_list_top20 = sorted(pangerank_list, reverse=True)[:21]
common_top = list(set(indegree_list_top20) & set(pangerank_list_top20))

print(common_top)

# nx.number_of_edges(emotion_subgraph) / ((nx.number_of_nodes(emotion_subgraph)
#                                          * (nx.number_of_nodes(emotion_subgraph) - 1) / 2))
# g_emotion_random = nx.erdos_renyi_graph(n=nx.number_of_nodes(
#     emotion_subgraph), p=0.06, seed=1234, directed=True)
# for (u, v) in g_emotion_random.edges():
#     g_emotion_random.edges[u, v]['weight'] = random.randint(0, 50)
# g_strong_random = nx.erdos_renyi_graph(n=nx.number_of_nodes(
#     g_cleaned_strong), p=0.02, seed=1234, directed=True)
# check network properties of main and emotion graph

# graphical characteristics


# Marcoscopic Level ------------------------------------------------------------
# here we examined the emotion graph's characteristics compared to the full graph
nx.density(g_cleaned_strong)
nx.density(emotion_subgraph)

main_ink_list = [value for (key, value) in g_cleaned_strong.in_degree()]
np.mean(main_ink_list)
np.std(main_ink_list)
subg_ink_list = [value for (key, value) in emotion_subgraph.in_degree()]
np.mean(subg_ink_list)
np.std(subg_ink_list)

main_outk_list = [value for (key, value) in g_cleaned_strong.out_degree()]
np.mean(main_outk_list)
np.std(main_outk_list)

subg_outk_list = [value for (key, value) in emotion_subgraph.out_degree()]
np.mean(subg_outk_list)
np.std(subg_outk_list)

nx.average_shortest_path_length(g_cleaned_strong)
nx.average_shortest_path_length(emotion_subgraph)

nx.average_clustering(g_cleaned_strong, weight="Weight")
nx.average_clustering(emotion_subgraph, weight="Weight")

nx.diameter(g_cleaned_strong)
nx.diameter(emotion_subgraph)

Pairs = pd.DataFrame(nx.all_pairs_shortest_path_length(emotion_subgraph))
np.mean(Pairs)
nx.average_shortest_path_length(emotion_subgraph)


# Mesoscopic Level -------------------------------------------------------------
# here we examined community structure of emotion graph

# get the communities using rb_pots (for weighted and directed graphs)
#
list_community = list()
list_communityitems = list()
for ith_iter in range(10000):
    random.seed(ith_iter)
    memberships = algorithms.rb_pots(emotion_subgraph, weights='Weight')
    list_communityitems.append(memberships.communities)
    n_community = len(memberships.communities)
    list_community.append(n_community)

count_table = {x: list_community.count(x) for x in list_community}
count_table

best_n_community = utils.most_frequent(list_community)
print(f'The optimal community solution: {best_n_community}')

# create color
pal = sns.color_palette('viridis', best_n_community)
pal_hex = pal.as_hex()

best_n_indices = [i for i in range(
    len(list_community)) if list_community[i] == best_n_community]

ref_list = list_communityitems[0]
ref_list_flatten = list(itertools.chain.from_iterable(ref_list))

membership_dataframe = pd.concat([pd.DataFrame(ref_list_flatten, columns=["node"]),
                                  pd.DataFrame(np.zeros((len(ref_list_flatten), best_n_community), dtype=np.int))],
                                 axis=1)
for i in best_n_indices[1:]:
    tmp_mat = np.zeros((best_n_community, best_n_community))
    tmp_list = list_communityitems[i]
    for j in range(best_n_community):
        for k in range(best_n_community):
            tmp_mat[j, k] = len(set(ref_list[j]) & set(
                tmp_list[k])) / float(len(set(ref_list[j]) | set(tmp_list[k])))
    new_order = list()
    for row in tmp_mat:
        new_order.append(np.argmax(row))
    if len(new_order) != best_n_community:
        next
    tmp_list_reordered = [tmp_list[i] for i in new_order]

    for ith_list in range(best_n_community):
        for ith_word in range(len(tmp_list_reordered[ith_list])):
            index = membership_dataframe.index[
                membership_dataframe['node'] ==
                tmp_list_reordered[ith_list][ith_word]
            ].tolist()[0]
            membership_dataframe.at[index, ith_list] = \
                membership_dataframe.at[index, ith_list] + 1

membership_dataframe_2 = membership_dataframe.set_index('node')

best_community = [[] for _ in range(best_n_community)]

for index, row in membership_dataframe_2.iterrows():
    best_community[np.argmax(row.values.tolist())].append(index)


# modularity
g_emotion_modularity = nx_comm.modularity(
    emotion_subgraph, best_community, weight='weight'
)

# permutate the edge weight to make sure our results are robust
list_modularity = list()
for ith_iter in range(10000):
    # general resampled graph
    _edgelist = nx.to_pandas_edgelist(emotion_subgraph)
    _edgelist['Weight'] = np.random.RandomState(
        seed=ith_iter
    ).permutation(
        _edgelist['Weight'].values
    )
    _emotion_resampled_subgraph = nx.from_pandas_edgelist(
        _edgelist,
        source='source',
        target='target',
        edge_attr='Weight',
        create_using=nx.DiGraph()
    )
    _memberships = algorithms.rb_pots(
        _emotion_resampled_subgraph,
        weights='Weight'
    )
    _modularity = nx_comm.modularity(
        _emotion_resampled_subgraph, _memberships.communities, weight='weight'
    )

    # store result
    list_modularity.append(_modularity)

perm_res = [x > g_emotion_modularity for x in list_modularity]

perm_p = np.sum(perm_res)/10000

# plot permutation plot
plt.figure(figsize=(10, 5))
plt.hist(list_modularity, bins=40, color='grey')
plt.axvline(x=g_emotion_modularity, color='red')
plt.text(0.24, 700, '$p_{permutation}$=0.0011', fontsize=12)
plt.ylabel('Frequency')
plt.xlabel('Modularity Q-value')
plt.savefig(Path('outputs', 'figs', 'modularity_permutation.pdf'))

# print emotion labels of each community
for i in range(best_n_community):
    print(list(set(best_community[i]) & set(emotion_stacked_list)))

# print top 10 non-emotion words
dict_emotion = dict(sorted(emotion_subgraph.in_degree,
                    key=lambda x: x[1], reverse=True))
for i in range(best_n_community):
    n = 0
    print(f'Community {i + 1}')
    for key in dict_emotion:
        if key in best_community[i]:
            print(key)
            n = n + 1
        if n == 10:
            break

# plot community graph
comm_graph_mem = dict()
for i in range(12):
    best_community_set = set(best_community[i])
    comm_graph_mem[f'{i}'] = best_community_set.intersection(
        set(emotion_stacked_list))

pal = sns.color_palette('viridis', best_n_community)
pal_hex = pal.as_hex()
community2color = dict(zip(range(best_n_community), pal_hex))

node2community = dict()
nodesize = dict()
nodelabel = dict()
nodealpha = dict()
i = 0
for community in best_community:
    for item in community:
        node2community[item] = i
        if item in common_top:
            print(item)
            nodesize[item] = 4
            nodealpha[item] = 0.85
            nodelabel[item] = item
        else:
            nodesize[item] = 2
            nodealpha[item] = 0.4
            nodelabel[item] = ""
    i = i + 1

node_color = {node: community2color[community_id]
              for node, community_id in node2community.items()}

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
Graph(emotion_subgraph,
      node_color=node_color,
      node_size=nodesize,
      node_edge_width=0,
      node_alpha=nodealpha,
      node_labels=nodelabel,
      node_label_fontdict=dict(size=10,
                               fontfamily='Source Han Sans HC',
                               fontweight='bold',
                               color="#FCFAF2",
                               alpha=0.8),
      edge_alpha=0.002,
      node_layout='community',
      node_layout_kwargs=dict(node_to_community=node2community),
      # edge_layout='bundled',
      edge_layout_kwargs=dict(k=10),
      )
fig.set_facecolor('#FCFAF2')
plt.savefig(Path('outputs', 'figs', 'cue-emotion_communities_subgraph.png'),
            dpi=300, bbox_inches='tight')


node_position = [g_emotion_nodes.index(x) for x in common_top]
node_sizes = list()
for ith in range(len(g_emotion_nodes)):
    if ith in node_position:
        node_sizes.append(500)
    else:
        node_sizes.append(10)

labels = {}
for node in emotion_subgraph.nodes():
    if node in common_top:
        # set the node name as the key and the label as its value
        labels[node] = node


node_pos = nx.spring_layout(emotion_subgraph, k=0.7, seed=1111)
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
nx.draw(G=emotion_subgraph,
        pos=node_pos,
        with_labels=False,
        node_size=node_sizes,
        arrowsize=3,
        font_color='#1C1C1C',
        node_color=membership_color,
        edge_color='#373C38',
        width=0.05,
        alpha=0.85,
        ax=ax)
nx.draw_networkx_labels(G=emotion_subgraph,
                        pos=node_pos,
                        labels=labels,
                        font_family='Source Han Sans HC',
                        font_size=10,
                        ax=ax)
fig.set_facecolor('#FCFAF2')
plt.savefig(Path('outputs', 'figs', 'cue-emotion_hubs_subgraph.png'),
            dpi=300, bbox_inches='tight')


emotional_words_pd = pd.read_csv(
    Path('outputs', 'tables', 'emotion_cues_rating.csv'), encoding='big5')


# Microscopic Level ------------------------------------------------------------
# here we examined some common pairs of synonymic emotion words in Cantonese
# including happiness, surprise and pathatic
subgraph_node = dict(happy1='開心',
                     happy2='快樂',
                     surprise1='驚訝',
                     surprise2='驚喜',
                     yumgung='陰公',
                     pity='可悲')

list_nodes = dict()

for key, value in subgraph_node.items():
    select_nodes = [node for node in g.neighbors(value)]
    select_nodes.append(value)
    sub_g = g.subgraph(select_nodes)
    selected_nodes2 = [node for node,
                       outdeg in sub_g.out_degree() if outdeg != 0]
    sub_g_cleaned = sub_g.subgraph(selected_nodes2)

    node_info = dict(sub_g_cleaned.in_degree())
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nx.draw(sub_g_cleaned, pos=nx.kamada_kawai_layout(sub_g_cleaned),
            font_family='Source Han Sans HC', with_labels=True,
            node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10,
            font_color='#1C1C1C', node_color='#F9BF45', edge_color='#373C38',
            width=0.5, alpha=0.9, ax=ax)
    fig.set_facecolor('#FCFAF2')
    plt.savefig(
        Path('outputs', 'figs', f'cue-{key}_subgraph.png'), dpi=400, bbox_inches='tight')

    list_nodes[value] = selected_nodes2

    averagerating = (emotional_words_pd
                     .query('cues == @selected_nodes2')
                     .mean(axis=0))
    sdrating = (emotional_words_pd
                .query('cues == @selected_nodes2')
                .std(axis=0))
    print(value)
    print(f'Mean: {averagerating}, SD: {sdrating}')

happy1_node_list = list_nodes.get('開心')
happy2_node_list = list_nodes.get('快樂')
surprise1_node_list = list_nodes.get('驚訝')
surprise2_node_list = list_nodes.get('驚喜')
pathatic1_node_list = list_nodes.get('陰公')
pathatic2_node_list = list_nodes.get('可悲')

happy_set_difference = list(set(happy1_node_list) ^ set(happy2_node_list))
happy1_unique_node = [i for i in happy_set_difference if i in happy1_node_list]
happy2_unique_node = [i for i in happy_set_difference if i in happy2_node_list]

surprise_set_difference = list(
    set(surprise1_node_list) ^ set(surprise2_node_list))
surprise1_unique_node = [
    i for i in surprise_set_difference if i in surprise1_node_list]
surprise2_unique_node = [
    i for i in surprise_set_difference if i in surprise2_node_list]

pathatic_set_difference = list(
    set(pathatic1_node_list) ^ set(pathatic2_node_list))
pathatic1_unique_node = [
    i for i in pathatic_set_difference if i in pathatic1_node_list]
pathatic2_unique_node = [
    i for i in pathatic_set_difference if i in pathatic2_node_list]

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

all_rating = (pd.concat([index, rating], axis=1)
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
    .drop([0, 1])
    .set_index('cues')
    .apply(pd.to_numeric, errors='coerce')
    .loc[:, ['valence_mean', 'arousal_mean', 'dominance_mean', 'concreteness_mean']])

happy1_nodes_rating = all_rating.query('cues == @happy1_node_list')
happy2_nodes_rating = all_rating.query('cues == @happy2_node_list')
surprise1_nodes_rating = all_rating.query(
    'cues == @surprise1_unique_node')
surprise2_nodes_rating = all_rating.query(
    'cues == @surprise2_unique_node')
pathatic1_nodes_rating = all_rating.query(
    'cues == @pathatic1_unique_node')
pathatic2_nodes_rating = all_rating.query(
    'cues == @pathatic2_unique_node')

cols = all_rating.columns
for col_name in cols:
    print(col_name)
    print("Happy")
    t, p = ttest_ind(
        happy1_nodes_rating[col_name], happy2_nodes_rating[col_name])
    print(researchpy.ttest(
        happy1_nodes_rating[col_name], happy2_nodes_rating[col_name]))
    print(f'{t}, {p}')
    print("Surprise")
    t, p = ttest_ind(
        surprise1_nodes_rating[col_name], surprise2_nodes_rating[col_name])
    print(researchpy.ttest(
        surprise1_nodes_rating[col_name], surprise2_nodes_rating[col_name]))
    print(f'{t}, {p}')
    print("Pathetic")
    t, p = ttest_ind(
        pathatic1_nodes_rating[col_name], pathatic2_nodes_rating[col_name])
    print(researchpy.ttest(
        pathatic1_nodes_rating[col_name], pathatic2_nodes_rating[col_name]))
    print(f'{t}, {p}')

# Equivalence Test
for col_name in cols:
    print(col_name)
    print("Happy")
    p, low_p, upp_p = ttost_ind(
        happy1_nodes_rating[col_name], happy2_nodes_rating[col_name],
        low=-0.5, upp=0.5
    )
    print(f'Lower: {low_p}, Upper: {upp_p}')
    print("Pathetic")
    p, low_p, upp_p = ttost_ind(
        pathatic1_nodes_rating[col_name], pathatic2_nodes_rating[col_name],
        low=-0.5, upp=0.5
    )
    print(f'Lower: {low_p}, Upper: {upp_p}')


graph_orig = nx.from_pandas_edgelist(df=r123_edges_emotions,
                                     source='cue',
                                     target='R123',
                                     edge_attr='Weight',
                                     create_using=nx.DiGraph())

g = graph_orig.copy()

n_edges = nx.number_of_edges(g)
n_nodes = nx.number_of_nodes(g)

# remove nodes that out degree is 0
selected_nodes = [node for node, outdeg in g.out_degree() if outdeg != 0]
g_cleaned = g.subgraph(selected_nodes)
n_edges = nx.number_of_edges(g_cleaned)
n_nodes = nx.number_of_nodes(g_cleaned)

deg_centrality = pd.DataFrame(list(nx.degree_centrality(
    g_cleaned).items()), columns=['cue', 'strength'])

fig = alt.Chart(deg_centrality).mark_bar().encode(
    x='strength:Q',
    y=alt.Y('cue:N', sort='-x')
)

fig.save(Path('outputs', 'figs', 'fig.html'))


node_info = dict(g_cleaned.out_degree())

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
nx.draw(g_cleaned, pos=nx.kamada_kawai_layout(g_cleaned),
        font_family='Source Han Sans HC', with_labels=True,
        node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10,
        font_color='#1C1C1C', node_color='#F9BF45', edge_color='#373C38',
        width=0.5, alpha=0.9, ax=ax)
fig.set_facecolor('#FCFAF2')
plt.savefig(Path('outputs', 'figs', 'emotionalwords_subgraph.png'),
            dpi=400, bbox_inches='tight')
