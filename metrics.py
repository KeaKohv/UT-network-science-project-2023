"""
Source of code: https://colab.research.google.com/drive/1KKugoFyUdydYC0XRyddcROzfQdMwDcnO?usp=sharing
That code was adapted to our purposes, e.g. reporting a wider range of @K.

"""

import torch
import numpy as np

from graph import get_user_positive_items


# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (intg): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                  for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


    # computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()



# wrapper function to get evaluation metrics
def get_metrics(model, edge_index, exclude_edge_indices, K):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple of dictionaries holding: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get interactions between every user and item - shape is num users x num movies
    interaction = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set interactions of excluded edges to large negative value
        interaction[exclude_users, exclude_items] = -(1 << 10)

        # get all unique users in evaluated split
        users = edge_index[0].unique()

        test_user_pos_items = get_user_positive_items(edge_index)

        # convert test user pos items dictionary into a list
        test_user_pos_items_list = [
            test_user_pos_items[user.item()] for user in users]

        recalls = {}
        precisions = {}
        ndcgs = {}
        for k in K:
            # get the top k recommended items for each user
            _, top_K_items = torch.topk(interaction, k=k)

            # determine the correctness of topk predictions
            r = []
            for user in users:
                ground_truth_items = test_user_pos_items[user.item()]
                label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
                r.append(label)
            r = torch.Tensor(np.array(r).astype('float'))

            recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
            recalls[k] = recall
            precisions[k] = precision
            ndcg = NDCGatK_r(test_user_pos_items_list, r, k)
            ndcgs[k] = ndcg

            print(f"[recall@{k}: {round(recall, 5)}, precision@{k}: {round(precision, 5)}, ndcg@{k}: {round(ndcg, 5)}")

    return recalls, precisions, ndcgs