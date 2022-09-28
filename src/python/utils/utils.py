from pathlib import Path
import pandas as pd
import numpy as np
import itertools


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def anova_table(aov):
    aov["mean_sq"] = aov[:]["sum_sq"]/aov[:]["df"]
    aov["eta_sq"] = aov[:-1]["sum_sq"]/sum(aov["sum_sq"])
    aov["omega_sq"] = (aov[:-1]["sum_sq"]-(aov[:-1]["df"] *
                       aov["mean_sq"][-1]))/(sum(aov["sum_sq"])+aov["mean_sq"][-1])
    cols = ["sum_sq", "df", "mean_sq", "F", "PR(>F)", "eta_sq", "omega_sq"]
    aov = aov[cols]
    return aov


def most_frequent(List):
    return max(set(List), key=List.count)


def reorder_community_members(ref_membership, best_n_indices, list_communityitems, n_community):
    ref_membership_flatten = list(
        itertools.chain.from_iterable(ref_membership))
    n_members = len(ref_membership_flatten)

    # create an empty dataframe for memberships
    memberships = pd.concat([pd.DataFrame(ref_membership_flatten, columns=["node"]),
                             pd.DataFrame(np.zeros((len(n_members), n_community),
                                                   dtype=np.int))], axis=1)

    for i in best_n_indices[1:]:
        tmp_mat = np.zeros((n_community, n_community))
        tmp_list = list_communityitems[i]
        for j in range(n_community):
            for k in range(n_community):
                tmp_mat[j, k] = len(set(ref_membership[j]) & set(
                    tmp_list[k])) / float(len(set(ref_membership[j]) | set(tmp_list[k])))
        new_order = list()
        for row in tmp_mat:
            new_order.append(np.argmax(row))
        if len(new_order) != n_community:
            next
        tmp_list_reordered = [tmp_list[i] for i in new_order]

        for ith_list in range(n_community):
            for ith_word in range(len(tmp_list_reordered[ith_list])):
                index = memberships.index[memberships['node'] == tmp_list_reordered[ith_list][ith_word]].tolist()[
                    0]
                memberships.at[index,
                               ith_list] = memberships.at[index, ith_list] + 1

    best_community = [[] for _ in range(n_community)]

    for index, row in memberships.iterrows():
        best_community[np.argmax(row.values.tolist())].append(index)

    return best_community
