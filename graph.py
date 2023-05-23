import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sentence_transformers import SentenceTransformer


def load_nodes(df, index_col, encoders=None):
    df_reset = df.set_index(index_col)

    mapping = {index: i for i, index in enumerate(df_reset.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df_reset[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edges(df, src_index_col, src_mapping, dst_index_col, dst_mapping):

    # So we can do the split by global timeline later
    df = df.sort_values('timestamp')

    edge_index = None
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]

    edge_timestamp = df.timestamp.values

    edge_attr = torch.from_numpy(df.rating.values).view(-1, 1).to(torch.long)
    edge_index = torch.tensor([src, dst])

    return edge_index, edge_attr, edge_timestamp


# Encoder for movie title
class SequenceEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()
    

# Encoder for other movie features
class CategoryEncoder:
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        categories = set(c for col in df.values for c in col.split(self.sep))
        mapping = {category: i for i, category in enumerate(categories)}

        x = torch.zeros(len(df), len(mapping))

        for i, col in enumerate(df.values):
            for category in col.split(self.sep):
                x[i, mapping[category]] = 1
        return x
    


# train/validation/test split
def train_val_test_split(edge_index, split_strategy='temporal'):
  num_interactions = edge_index.shape[1]

  # split by global timepoint, there is no data leakage
  # 80/10/10 ratio 
  if split_strategy == 'temporal':
    train_edge_index = edge_index[:,np.arange(0,num_interactions*0.8)]
    val_edge_index = edge_index[:,np.arange(num_interactions*0.8,num_interactions*0.9)]
    test_edge_index = edge_index[:,np.arange(num_interactions*0.9,num_interactions)]

  # Random split by interactions ratio, there is data leakage
  # 80/10/10 ratio
  elif split_strategy == 'random':
    all_indices = [i for i in range(num_interactions)]

    train_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=1)
    val_indices, test_indices = train_test_split(
        test_indices, test_size=0.5, random_state=1)

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

  # Leave-one-out strategy, there is data leakage
  # Leave the latest interaction for test, one before the latest for valid, rest for train
  # NB proportsioonid tulevad teised (mitte 80/10/10), kuna siin pole ratio j2rgi split, vaid userite kaupa
  elif split_strategy == 'leave_one_out':
    unique_ids = np.unique(edge_index[0].cpu().numpy())
    edges = edge_index.cpu().numpy()
    train_edge_index = [[],[]]
    val_edge_index = [[],[]]
    test_edge_index = [[],[]]

    for id in unique_ids:
      user_edges = np.vstack((edges[0][edges[0]==id], edges[1][edges[0]==id]))

      # Check the number of interactions for each user, if it is less than three, then only treat the interaction as train or train and val
      if len(user_edges[0])==1:
        train_edge_index = np.concatenate((train_edge_index,user_edges),axis=1).astype(int)

      elif len(user_edges[0])==2:
        train_edge_index = np.concatenate((train_edge_index,user_edges[:,:-1]),axis=1).astype(int)
        test_edge_index = np.concatenate((test_edge_index,user_edges[:,-1].reshape(-1,1)),axis=1).astype(int)

      else:    
        test_edge_index = np.concatenate((test_edge_index,user_edges[:,-1].reshape(-1,1)),axis=1).astype(int)
        val_edge_index = np.concatenate((val_edge_index,user_edges[:,-2].reshape(-1,1)),axis=1).astype(int)
        train_edge_index = np.concatenate((train_edge_index,user_edges[:,:-2]),axis=1).astype(int)

    train_edge_index = torch.tensor(train_edge_index)
    test_edge_index = torch.tensor(test_edge_index)
    val_edge_index = torch.tensor(val_edge_index)


  return train_edge_index, val_edge_index, test_edge_index


# helper function to get N_u
def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items