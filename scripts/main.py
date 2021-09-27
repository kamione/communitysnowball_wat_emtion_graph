import matplotlib.font_manager as fm
import os
import altair
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import Counter
from matplotlib import font_manager
from src.python.core import wcloud

# Add font family: Source Han Sans HC
font_files = font_manager.findSystemFonts(fontpaths=str(Path("fonts")))
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# check font
# font_set = {f.name for f in font_manager.fontManager.ttflist}
# for f in font_set:
#    print(f)
# [f for f in fm.fontManager.ttflist if 'Source Han Sans HC' in f.name]

# seaborn plotting environment
sns.set_style("whitegrid")
sns.set(font="Source Han Sans HC")  # for Chinese characters

# Data I/O ---------------------------------------------------------------------
data_dir = Path("data", "raw", "20210514_snowball_cleaned_chinese_only.csv")
data = pd.read_csv(data_dir, dtype='unicode')

data_cleaned = (data
                .assign(
                    created_at=data["created_at"].apply(
                        lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M")),
                    created_year=data["created_at"].apply(
                        lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").year),
                    created_month=data["created_at"].apply(
                        lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M").month))
                )


# Data Visualization -----------------------------------------------------------
# all time frame
texts = data_cleaned["cue"].tolist()
texts = ' '.join(texts)

# 2014
texts_2014 = data_cleaned.query("created_year == 2014")["cue"].tolist()
texts_2014 = ' '.join(texts_2014)
wcloud.MyWordCloud(texts_2014, "2014").plot()
# count the frequency
data_cleaned.query("created_year == 2014")["cue"].value_counts()

# 2015
texts_2015 = data_cleaned.query("created_year == 2015")["cue"].tolist()
texts_2015 = ' '.join(texts_2015)
wcloud.MyWordCloud(texts_2015, "2015").plot()
data_cleaned.query("created_year == 2015")["cue"].value_counts()

# 2016
texts_2016 = data_cleaned.query("created_year == 2016")["cue"].tolist()
texts_2016 = ' '.join(texts_2016)
wcloud.MyWordCloud(texts_2016, "2016").plot()
data_cleaned.query("created_year == 2016")["cue"].value_counts()

# 2017
texts_2017 = data_cleaned.query("created_year == 2017")["cue"].tolist()
texts_2017 = ' '.join(texts_2016)
wcloud.MyWordCloud(texts_2017, "2017").plot()
data_cleaned.query("created_year == 2017")["cue"].value_counts()

# 2014 to 2015
texts_early = data_cleaned.query("created_year == 2014|created_year == 2015")[
    "cue"].tolist()
texts_early = ' '.join(texts_early)

# 2020
texts_recent = data_cleaned.query("created_year == 2020")["cue"].tolist()
texts_recent = ' '.join(texts_recent)

data_cleaned.query("created_year == 2020")["cue"].value_counts()

# plot word cloud of cues
wcloud.MyWordCloud(texts, "all").plot()
wcloud.MyWordCloud(texts_early, "2014to15").plot()
wcloud.MyWordCloud(texts_recent, "2020").plot()


np.intersect1d(data_cleaned.query("created_year == 2014")["cue"].unique(),
               data_cleaned.query("created_year == 2015")["cue"].unique()).shape
np.intersect1d(data_cleaned.query("created_year == 2015")["cue"].unique(),
               data_cleaned.query("created_year == 2016")["cue"].unique()).shape
np.intersect1d(data_cleaned.query("created_year == 2016")["cue"].unique(),
               data_cleaned.query("created_year == 2020")["cue"].unique()).shape
np.intersect1d(data_cleaned.query("created_year == 2016")["cue"].unique(),
               data_cleaned.query("created_year == 2017")["cue"].unique()).shape
np.intersect1d(data_cleaned.query("created_year == 2017")["cue"].unique(),
               data_cleaned.query("created_year == 2020")["cue"].unique()).shape
np.intersect1d(data_cleaned.query("created_year == 2014")["cue"].unique(),
               data_cleaned.query("created_year == 2016")["cue"].unique()).shape


# plot totoal numbers of responses across years
fig = sns.catplot(y="created_year", kind="count", palette="ch:.25",
                  data=data_cleaned, height=3, aspect=1.5)
fig.set(xlabel="Total Numbers of Responses", ylabel="Year")
fig.savefig(Path("outputs", "figs", "year_resp_chart.pdf"))


# Network Analysis -------------------------------------------------------------
# R123: Combine Response R1, R2, and R3
r123_wide = data_cleaned.filter(regex="created_year|cue|r1|r2|r3")
r123_long = pd.melt(r123_wide, id_vars=["cue", "created_year"],
                    value_vars=["r1", "r2", "r3"], value_name="R123")
r123_long = r123_long.drop(r123_long[r123_long.R123 == '沒有更多回應'].index)


np.intersect1d(r123_long.query("created_year == 2014")["R123"].astype(str).unique(),
               r123_long.query("created_year == 2015")["R123"].astype(str).unique()).shape
np.intersect1d(r123_long.query("created_year == 2015")["R123"].astype(str).unique(),
               r123_long.query("created_year == 2016")["R123"].astype(str).unique()).shape
np.intersect1d(r123_long.query("created_year == 2016")["R123"].astype(str).unique(),
               r123_long.query("created_year == 2020")["R123"].astype(str).unique()).shape
np.intersect1d(r123_long.query("created_year == 2014")["R123"].astype(str).unique(),
               r123_long.query("created_year == 2020")["R123"].astype(str).unique()).shape
np.intersect1d(r123_long.query("created_year == 2015")["R123"].astype(str).unique(),
               r123_long.query("created_year == 2020")["R123"].astype(str).unique()).shape
np.intersect1d(r123_long.query("created_year == 2014")["R123"].astype(str).unique(),
               r123_long.query("created_year == 2016")["R123"].astype(str).unique()).shape


r123_edges = (r123_long
              .groupby(["cue", "R123"])
              .agg({"R123": "size"})
              .rename({"R123": "Freq"}, axis=1)
              .reset_index()
              )

r123_edges_early = (
    r123_long
    .query("created_year==2014|created_year==2015")
    .groupby(["cue", "R123"])
    .agg({"R123": "size"})
    .rename({"R123": "Freq"}, axis=1)
    .reset_index()
)

r123_edges_recent = (
    r123_long
    .query("created_year==2020")
    .groupby(["cue", "R123"])
    .agg({"R123": "size"})
    .rename({"R123": "Freq"}, axis=1)
    .reset_index()
)

####################################################
# create Graph (all time points) using networkx
####################################################
graph_orig = nx.from_pandas_edgelist(df=r123_edges, source="cue", target="R123",
                                     edge_attr="Freq",
                                     create_using=nx.DiGraph())

g = graph_orig.copy()

n_edges = nx.number_of_edges(g)
n_nodes = nx.number_of_nodes(g)

# list(g.neighbors("警察"))
# list(g.successors("警察"))
# list(g.predecessors("警察"))

# remove nodes that out degree is 0
selected_nodes = [node for node, outdeg in g.out_degree() if outdeg != 0]
subg = g.subgraph(selected_nodes)
n_edges = nx.number_of_edges(subg)
n_nodes = nx.number_of_nodes(subg)

subgraph_node = dict(student="學生", police="警察", gov="政府", idiot="傻仔",
                     fear="驚")

for key, value in subgraph_node.items():
    select_nodes = [node for node in g.neighbors(value)]
    select_nodes.append(value)
    police_g = g.subgraph(select_nodes)
    selected_nodes2 = [node for node,
                       outdeg in police_g.out_degree() if outdeg != 0]
    police_g_cleaned = police_g.subgraph(selected_nodes2)
    node_info = dict(police_g_cleaned.out_degree())

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nx.draw(police_g_cleaned, pos=nx.kamada_kawai_layout(police_g_cleaned),
            font_family="Source Han Sans HC", with_labels=True,
            node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10, font_color="#1C1C1C",
            node_color="#F9BF45", edge_color="#373C38", width=0.5, alpha=0.9, ax=ax)
    fig.set_facecolor("#FCFAF2")
    plt.savefig(
        Path("output", "figs", f"{key}_subgraph.png"), dpi=400, bbox_inches="tight")


####################################################
# create Graph (2014/2015) using networkx
####################################################
graph_early_orig = nx.from_pandas_edgelist(df=r123_edges_early, source="cue",
                                           target="R123",
                                           edge_attr="Freq",
                                           create_using=nx.DiGraph())
g_early = graph_early_orig.copy()

n_edges = nx.number_of_edges(g_early)
n_nodes = nx.number_of_nodes(g_early)
print(nx.info(g_early))

# remove nodes that out degree is 0
selected_nodes = [node for node, outdeg in g_early.out_degree() if outdeg != 0]
subg_early = g_early.subgraph(selected_nodes)
n_edges = nx.number_of_edges(subg_early)
n_nodes = nx.number_of_nodes(subg_early)
print(nx.info(subg_early))
subgraph_node = dict(student="學生", weapon="武器", police="警察")

for key, value in subgraph_node.items():
    select_nodes = [node for node in g_early.neighbors(value)]
    select_nodes.append(value)
    sub_g = g_early.subgraph(select_nodes)
    selected_nodes2 = [node for node,
                       outdeg in sub_g.out_degree() if outdeg != 0]
    sub_g_cleaned = sub_g.subgraph(selected_nodes2)
    node_info = dict(sub_g_cleaned.out_degree())

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nx.draw(sub_g_cleaned, pos=nx.kamada_kawai_layout(sub_g_cleaned),
            font_family="Source Han Sans HC", with_labels=True,
            node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10, font_color="#1C1C1C",
            node_color="#F9BF45", edge_color="#373C38", width=0.5, alpha=0.9, ax=ax)
    fig.set_facecolor("#FCFAF2")
    plt.savefig(Path("outputs", "figs",
                f"{key}_subgraph_early.png"), dpi=400, bbox_inches="tight")


####################################################
# create Graph (2020) using networkx
####################################################
graph_recent_orig = nx.from_pandas_edgelist(df=r123_edges_recent, source="cue",
                                            target="R123",
                                            edge_attr="Freq",
                                            create_using=nx.DiGraph())
graph_recent = graph_recent_orig.copy()

n_edges = nx.number_of_edges(graph_recent)
n_nodes = nx.number_of_nodes(graph_recent)
print(nx.info(graph_recent))

# remove nodes that out degree is 0
selected_nodes = [node for node,
                  outdeg in graph_recent.out_degree() if outdeg != 0]
subg_recent = graph_recent.subgraph(selected_nodes)
n_edges = nx.number_of_edges(subg_recent)
n_nodes = nx.number_of_nodes(subg_recent)
print(nx.info(subg_recent))
subgraph_node = dict(wuhanvirus="新冠病毒", word="生字")

for key, value in subgraph_node.items():
    select_nodes = [node for node in graph_recent.neighbors(value)]
    select_nodes.append(value)
    sub_g = graph_recent.subgraph(select_nodes)
    selected_nodes2 = [node for node,
                       outdeg in sub_g.out_degree() if outdeg != 0]
    sub_g_cleaned = sub_g.subgraph(selected_nodes2)
    node_info = dict(sub_g_cleaned.out_degree())

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nx.draw(sub_g_cleaned, pos=nx.kamada_kawai_layout(sub_g_cleaned),
            font_family="Source Han Sans HC", with_labels=True,
            node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10, font_color="#1C1C1C",
            node_color="#F9BF45", edge_color="#373C38", width=0.5, alpha=0.9, ax=ax)
    fig.set_facecolor("#FCFAF2")
    plt.savefig(Path("outputs", "figs",
                f"{key}_subgraph_recent.png"), dpi=400, bbox_inches="tight")


node_early = list(subg_early.nodes)
node_recent = list(subg_recent.nodes)

list(set(node_early) & set(node_recent))


# Update 20210517


data_cleaned.age = data_cleaned.age.astype("int32")
# create age group
data_cleaned["age_group"] = pd.cut(
    x=data_cleaned.age,
    bins=[0, 24, 60, 100],
    labels=["Adolescent", "Adults", "Elderly"]
)


df_2014to16_adolescent = (
    data_cleaned
    .query("created_year in [2014, 2015, 2016]")
    .query("age_group == 'Adolescent'"))
df_2014to16_adult = (
    data_cleaned
    .query("created_year in [2014, 2015, 2016]")
    .query("age_group == 'Adults'"))
df_2014to16_elderly = (
    data_cleaned
    .query("created_year in [2014, 2015, 2016]")
    .query("age_group == 'Elderly'"))
df_2014to16_male = (
    data_cleaned
    .query("created_year in [2014, 2015, 2016]")
    .query("gender == 'ma'"))
df_2014to16_female = (
    data_cleaned
    .query("created_year in [2014, 2015, 2016]")
    .query("gender == 'fe'"))


#
texts_adolescent = df_2014to16_adolescent["cue"].tolist()
texts_adolescent = ' '.join(texts_adolescent)
wcloud.MyWordCloud(texts_adolescent, "adolescent").plot()

df_2014to16_adolescent["cue"].value_counts().head(10)

texts_adult = df_2014to16_adult["cue"].tolist()
texts_adult = ' '.join(texts_adult)
wcloud.MyWordCloud(texts_adult, "adult").plot()

df_2014to16_adult["cue"].value_counts().head(10)

# elderly
texts_elderly = df_2014to16_elderly["cue"].tolist()
texts_elderly = ' '.join(texts_elderly)
wcloud.MyWordCloud(texts_elderly, "elderly").plot()

df_2014to16_elderly["cue"].value_counts().head(10)


# male
texts_male = df_2014to16_male["cue"].tolist()
texts_male = ' '.join(texts_male)
wcloud.MyWordCloud(texts_male, "male").plot()

df_2014to16_male["cue"].value_counts().head(10)

# female
texts_female = df_2014to16_female["cue"].tolist()
texts_female = ' '.join(texts_female)
wcloud.MyWordCloud(texts_female, "female").plot()

df_2014to16_female["cue"].value_counts().head(10)


# build graph
r123_wide = df_2014to16_adolescent.filter(regex="cue|r1|r2|r3")
r123_long = pd.melt(r123_wide, id_vars=["cue"],
                    value_vars=["r1", "r2", "r3"], value_name="R123")
r123_edges_adolescent = (
    r123_long
    .groupby(["cue", "R123"])
    .agg({"R123": "size"})
    .rename({"R123": "Freq"}, axis=1)
    .reset_index()
)


####################################################
# create Graph (2020) using networkx
####################################################
graph_adolescent_orig = nx.from_pandas_edgelist(
    df=r123_edges_adolescent, source="cue", target="R123", edge_attr="Freq",
    create_using=nx.DiGraph())
graph_adolescent = graph_adolescent_orig.copy()


n_edges = nx.number_of_edges(graph_adolescent)
n_nodes = nx.number_of_nodes(graph_adolescent)
print(nx.info(graph_adolescent))

# remove nodes that out degree is 0
selected_nodes = [node for node,
                  outdeg in graph_adolescent.out_degree() if outdeg != 0]
subg_adolescent = graph_adolescent.subgraph(selected_nodes)
n_edges = nx.number_of_edges(subg_adolescent)
n_nodes = nx.number_of_nodes(subg_adolescent)
print(nx.info(subg_adolescent))
subgraph_node = dict(student="學生")

for key, value in subgraph_node.items():
    select_nodes = [node for node in graph_adolescent.neighbors(value)]
    select_nodes.append(value)
    sub_g = graph_adolescent.subgraph(select_nodes)
    selected_nodes2 = [node for node,
                       outdeg in sub_g.out_degree() if outdeg != 0]
    sub_g_cleaned = sub_g.subgraph(selected_nodes2)
    node_info = dict(sub_g_cleaned.out_degree())

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nx.draw(sub_g_cleaned, pos=nx.kamada_kawai_layout(sub_g_cleaned),
            font_family="Source Han Sans HC", with_labels=True,
            node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10, font_color="#1C1C1C",
            node_color="#F9BF45", edge_color="#373C38", width=0.5, alpha=0.9, ax=ax)
    fig.set_facecolor("#FCFAF2")
    plt.savefig(Path("outputs", "figs",
                f"{key}_subgraph_adolescent.png"), dpi=400, bbox_inches="tight")


####################################################
# create adult subgroup using networkx
####################################################

r123_wide = df_2014to16_adult.filter(regex="cue|r1|r2|r3")
r123_long = pd.melt(r123_wide, id_vars=["cue"],
                    value_vars=["r1", "r2", "r3"], value_name="R123")
r123_edges_adult = (
    r123_long
    .groupby(["cue", "R123"])
    .agg({"R123": "size"})
    .rename({"R123": "Freq"}, axis=1)
    .reset_index()
)
graph_adult_orig = nx.from_pandas_edgelist(
    df=r123_edges_adult, source="cue", target="R123", edge_attr="Freq",
    create_using=nx.DiGraph())
graph_adult = graph_adult_orig.copy()


n_edges = nx.number_of_edges(graph_adult)
n_nodes = nx.number_of_nodes(graph_adult)
print(nx.info(graph_adult))

# remove nodes that out degree is 0
selected_nodes = [node for node,
                  outdeg in graph_adult.out_degree() if outdeg != 0]
subg_adult = graph_adult.subgraph(selected_nodes)
n_edges = nx.number_of_edges(subg_adult)
n_nodes = nx.number_of_nodes(subg_adult)
print(nx.info(subg_adult))
subgraph_node = dict(student="學生")

for key, value in subgraph_node.items():
    select_nodes = [node for node in graph_adult.neighbors(value)]
    select_nodes.append(value)
    sub_g = graph_adult.subgraph(select_nodes)
    selected_nodes2 = [node for node,
                       outdeg in sub_g.out_degree() if outdeg != 0]
    sub_g_cleaned = sub_g.subgraph(selected_nodes2)
    node_info = dict(sub_g_cleaned.out_degree())

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nx.draw(sub_g_cleaned, pos=nx.kamada_kawai_layout(sub_g_cleaned),
            font_family="Source Han Sans HC", with_labels=True,
            node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10, font_color="#1C1C1C",
            node_color="#F9BF45", edge_color="#373C38", width=0.5, alpha=0.9, ax=ax)
    fig.set_facecolor("#FCFAF2")
    plt.savefig(Path("outputs", "figs",
                f"{key}_subgraph_adult.png"), dpi=400, bbox_inches="tight")


####################################################
# create male subgroup using networkx
####################################################

r123_wide = df_2014to16_male.filter(regex="cue|r1|r2|r3")
r123_long = pd.melt(r123_wide, id_vars=["cue"],
                    value_vars=["r1", "r2", "r3"], value_name="R123")
r123_edges_male = (
    r123_long
    .groupby(["cue", "R123"])
    .agg({"R123": "size"})
    .rename({"R123": "Freq"}, axis=1)
    .reset_index()
)
graph_male_orig = nx.from_pandas_edgelist(
    df=r123_edges_male, source="cue", target="R123", edge_attr="Freq",
    create_using=nx.DiGraph())
graph_male = graph_male_orig.copy()


n_edges = nx.number_of_edges(graph_male)
n_nodes = nx.number_of_nodes(graph_male)
print(nx.info(graph_male))

# remove nodes that out degree is 0
selected_nodes = [node for node,
                  outdeg in graph_male.out_degree() if outdeg != 0]
subg_male = graph_male.subgraph(selected_nodes)
n_edges = nx.number_of_edges(subg_male)
n_nodes = nx.number_of_nodes(subg_male)
print(nx.info(subg_male))
subgraph_node = dict(student="學生")

for key, value in subgraph_node.items():
    select_nodes = [node for node in graph_male.neighbors(value)]
    select_nodes.append(value)
    sub_g = graph_male.subgraph(select_nodes)
    selected_nodes2 = [node for node,
                       outdeg in sub_g.out_degree() if outdeg != 0]
    sub_g_cleaned = sub_g.subgraph(selected_nodes2)
    node_info = dict(sub_g_cleaned.out_degree())

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nx.draw(sub_g_cleaned, pos=nx.kamada_kawai_layout(sub_g_cleaned),
            font_family="Source Han Sans HC", with_labels=True,
            node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10, font_color="#1C1C1C",
            node_color="#F9BF45", edge_color="#373C38", width=0.5, alpha=0.9, ax=ax)
    fig.set_facecolor("#FCFAF2")
    plt.savefig(Path("outputs", "figs",
                f"{key}_subgraph_male.png"), dpi=400, bbox_inches="tight")


####################################################
# create female subgroup using networkx
####################################################

r123_wide = df_2014to16_female.filter(regex="cue|r1|r2|r3")
r123_long = pd.melt(r123_wide, id_vars=["cue"],
                    value_vars=["r1", "r2", "r3"], value_name="R123")
r123_edges_female = (
    r123_long
    .groupby(["cue", "R123"])
    .agg({"R123": "size"})
    .rename({"R123": "Freq"}, axis=1)
    .reset_index()
)
graph_female_orig = nx.from_pandas_edgelist(
    df=r123_edges_female, source="cue", target="R123", edge_attr="Freq",
    create_using=nx.DiGraph())
graph_female = graph_female_orig.copy()


n_edges = nx.number_of_edges(graph_female)
n_nodes = nx.number_of_nodes(graph_female)
print(nx.info(graph_female))

# remove nodes that out degree is 0
selected_nodes = [node for node,
                  outdeg in graph_female.out_degree() if outdeg != 0]
subg_female = graph_female.subgraph(selected_nodes)
n_edges = nx.number_of_edges(subg_female)
n_nodes = nx.number_of_nodes(subg_female)
print(nx.info(subg_female))
subgraph_node = dict(student="學生")

for key, value in subgraph_node.items():
    select_nodes = [node for node in graph_female.neighbors(value)]
    select_nodes.append(value)
    sub_g = graph_female.subgraph(select_nodes)
    selected_nodes2 = [node for node,
                       outdeg in sub_g.out_degree() if outdeg != 0]
    sub_g_cleaned = sub_g.subgraph(selected_nodes2)
    node_info = dict(sub_g_cleaned.out_degree())

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nx.draw(sub_g_cleaned, pos=nx.kamada_kawai_layout(sub_g_cleaned),
            font_family="Source Han Sans HC", with_labels=True,
            node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10, font_color="#1C1C1C",
            node_color="#F9BF45", edge_color="#373C38", width=0.5, alpha=0.9, ax=ax)
    fig.set_facecolor("#FCFAF2")
    plt.savefig(Path("outputs", "figs",
                f"{key}_subgraph_female.png"), dpi=400, bbox_inches="tight")


# get R1 as cue and R2 | R3 as responses

# R123: Combine Response R1, R2, and R3
r123_wide = data_cleaned.filter(regex="created_year|cue|r1|r2|r3")

r23_long = pd.melt(r123_wide, id_vars=["cue", "r1", "created_year"],
                   value_vars=["r2", "r3"], value_name="r23")
r23_long = r23_long.drop(r23_long[r23_long.r23 == '沒有更多回應'].index)


top20 = r23_long.query("r1 == '工作'").r23.value_counts().head(
    20).rename_axis('unique_values').reset_index(name='counts')

plt.figure(figsize=(10, 5))
fig = sns.barplot(x='unique_values', y='counts',
                  data=top20, alpha=0.8, palette="rocket")
fig.set(title="R1: 工作", xlabel="R2 & R3", ylabel="Frequency")
plt.savefig(Path("outputs", "figs", "r1_work_r23_frequency_barplot.pdf"))


top20 = (r23_long
         .query("r1 == '工作'")
         .query("created_year==2014")
         .r23
         .value_counts()
         .head(20)
         .rename_axis('unique_values')
         .reset_index(name='counts'))

plt.figure(figsize=(10, 5))
fig = sns.barplot(x='unique_values', y='counts',
                  data=top20, alpha=0.8, palette="rocket")
fig.set(title="R1: 工作", xlabel="R2 & R3", ylabel="Frequency")
plt.savefig(Path("outputs", "figs", "2014_r1_work_r23_frequency_barplot.pdf"))






# Emotional Words
# Network Analysis -------------------------------------------------------------
# R123: Combine Response R1, R2, and R3
r123_wide = data_cleaned.filter(regex="created_year|cue|r1|r2|r3")
r123_long = pd.melt(r123_wide, id_vars=["cue", "created_year"],
                    value_vars=["r1", "r2", "r3"], value_name="R123")
r123_long = r123_long.drop(r123_long[r123_long.R123 == '沒有更多回應'].index)

r123_long.query("cue == '高興'")["created_year"].value_counts()

# create edges dataframe
r123_edges = (r123_long
              .groupby(["cue", "R123"])
              .agg({"R123": "size"})
              .rename({"R123": "Freq"}, axis=1)
              .reset_index()
              )


graph_orig = nx.from_pandas_edgelist(df=r123_edges, source="cue", target="R123",
                                     edge_attr="Freq",
                                     create_using=nx.DiGraph())

g = graph_orig.copy()

n_edges = nx.number_of_edges(g)
n_nodes = nx.number_of_nodes(g)

# list(g.neighbors("警察"))
# list(g.successors("警察"))
# list(g.predecessors("警察"))

# remove nodes that out degree is 0
selected_nodes = [node for node, outdeg in g.out_degree() if outdeg != 0]
subg = g.subgraph(selected_nodes)
n_edges = nx.number_of_edges(subg)
n_nodes = nx.number_of_nodes(subg)

subgraph_node = dict(happy1="高興", happy2="快樂", regret="後悔", fear="驚")

for key, value in subgraph_node.items():
    select_nodes = [node for node in g.neighbors(value)]
    select_nodes.append(value)
    sub_g = g.subgraph(select_nodes)
    selected_nodes2 = [node for node,
                       outdeg in sub_g.out_degree() if outdeg != 0]
    sub_g_cleaned = sub_g.subgraph(selected_nodes2)
    node_info = dict(sub_g_cleaned.out_degree())

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    nx.draw(sub_g_cleaned, pos=nx.kamada_kawai_layout(sub_g_cleaned),
            font_family="Source Han Sans HC", with_labels=True,
            node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10,
            font_color="#1C1C1C", node_color="#F9BF45", edge_color="#373C38",
            width=0.5, alpha=0.9, ax=ax)
    fig.set_facecolor("#FCFAF2")
    plt.savefig(
        Path("outputs", "figs", f"{key}_subgraph.png"), dpi=400, bbox_inches="tight")




# create edges dataframe only from emotional words
emotional_words_list=["高興", "快樂", "愉快", "喜悅", "歡樂", "開心", "緊張", "驚", 
                      "後悔", "失落", "暴躁", "空虛", "孤獨", "焦慮", "擔心", "煩厭",
                      "生氣", "孤單", "失望", "氣憤", "內疚", "擔憂", "驚慌", "痛楚",
                      "憤怒", "衰傷", "寂寞", "悲慘", "痛心", "傷心", "嫉妒", "沮喪",
                      "仇恨", "心痛", "唔開心", "不開心"]

r123_edges_emotions = (r123_long
                       .groupby(["cue", "R123"])
                       .agg({"R123": "size"})
                       .rename({"R123": "Freq"}, axis=1)
                       .reset_index()
                       .query("cue == @emotional_words_list")
                       )

count = r123_edges_emotions["cue"].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(count.index, count.values, alpha=0.8, palette="Blues_d")
plt.title("Frequency of Cues")
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel("Emotional Words", fontsize=12)
plt.xticks(rotation=70)
plt.savefig(Path("outputs", "figs", "emotionalwords_frequency.png"), dpi=400, bbox_inches="tight")



graph_orig = nx.from_pandas_edgelist(df=r123_edges_emotions,
                                     source="cue",
                                     target="R123",
                                     edge_attr="Freq",
                                     create_using=nx.DiGraph())

g = graph_orig.copy()

n_edges = nx.number_of_edges(g)
n_nodes = nx.number_of_nodes(g)

# remove nodes that out degree is 0
selected_nodes = [node for node, outdeg in g.out_degree() if outdeg != 0]
subg = g.subgraph(selected_nodes)
n_edges = nx.number_of_edges(subg)
n_nodes = nx.number_of_nodes(subg)

deg_centrality = pd.DataFrame(list(nx.degree_centrality(subg).items()), columns=['cue', 'strength'])

fig = altair.Chart(deg_centrality).mark_bar().encode(
    x='strength:Q',
    y=altair.Y('cue:N', sort='-x')
)

fig.save(Path("outputs", "figs", "fig.html"))



node_info = dict(subg.out_degree())

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
nx.draw(subg, pos=nx.kamada_kawai_layout(subg),
        font_family="Source Han Sans HC", with_labels=True,
        node_size=[(v + 1) * 85 for v in node_info.values()], font_size=10,
        font_color="#1C1C1C", node_color="#F9BF45", edge_color="#373C38",
        width=0.5, alpha=0.9, ax=ax)
fig.set_facecolor("#FCFAF2")
plt.savefig(Path("outputs", "figs", "emotionalwords_subgraph.png"), dpi=400, bbox_inches="tight")