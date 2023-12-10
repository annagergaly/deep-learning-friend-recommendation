import os

import numpy as np
import torch
import pandas as pd

import torch
import torch_geometric.datasets as datasets
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.datasets.snap_dataset import EgoData
from torch_geometric.data.makedirs import makedirs
from torch_geometric.utils import coalesce
from transformers import AutoTokenizer, AutoModel
import os.path as osp
from typing import Any, Callable, List, Optional
from torch_geometric.data import (
    Data,
    Dataset,
    InMemoryDataset,
    download_url,
    extract_gz,
    extract_tar
)

import sys
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


MODEL_FOR_EMBEDDINGS = "nlptown/bert-base-multilingual-uncased-sentiment"
CLUSTER_PATH = "twitter_featurenames_clusters"

def read_ego_feats_as_diff(egofeat_file, feat_file, featnames_file, feature_types):
    try:
        x_ego = pd.read_csv(egofeat_file, sep=' ', header=None,dtype=np.float32)
        x_ego = torch.from_numpy(x_ego.values)

        x = pd.read_csv(feat_file, sep=' ', header=None, dtype=np.float32)
        x = torch.from_numpy(x.values)[:, 1:]

        x_all = torch.cat([x, x_ego], dim=0)
    except Exception:
        #If there is an error during reading, that means that the files were bad
        #E.g. the egofeat or the feat files are empty
        return None

    #Gather all feature names
    with open(featnames_file, 'r') as f:
        featnames = f.read().split('\n')[:-1]
        featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
    featnames = sorted(featnames)

    #Gather the indices of the features which start with the given feature type
    #for each feature type
    idx = []
    for feature_type in feature_types:
        try:
            idx.append([x[:len(feature_type)] for x in featnames].index(feature_type))
        except Exception:
            idx.append(-1)

    #Difference compared to the ego (simple binary multi-hot vectors are multiplied)
    intersect = x_all * x_ego

    #Value corresponding to each feature type will be the number of shared features
    #between the ego and a node in that given feature_type
    cats = []
    for i in range(len(idx)):
        if idx[i] == -1:
            cats.append(torch.zeros((intersect.shape[0])))
            continue
        if i == (len(idx) -1):
            cats.append(intersect[:, idx[i]:].sum(axis=1))
        else:
            cats.append(intersect[:, idx[i]:idx[i+1]].sum(axis=1))

    return torch.stack(cats, axis=1)
    

def ego_read_all_featnames(files: List[str]):
    #Read all feature names from the featnames files and sort them
    files = [
        x for x in files if x.split('.')[-1] in
        ['featnames']
    ]
    files = sorted(files)
    all_featnames = []
    for i in range(len(files)):
        featnames_file = files[i]
        with open(featnames_file, 'r') as f:
            featnames = f.read().split('\n')[:-1]
            featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
            all_featnames += featnames
    all_featnames = sorted(list(set(all_featnames)))
    return all_featnames
  
def cluster_featnames(all_featnames):
    #If the cluster file exists, then return that
    if os.path.exists(CLUSTER_PATH):
        with open(CLUSTER_PATH, "rb") as f:
            clusters = np.load(f)
            return clusters
    #Create embedding for each feature name and then apply KMeans clustering
    if torch.cuda.is_available():
        device="cuda:0"
    else:
        device="cpu"
    model = AutoModel.from_pretrained(MODEL_FOR_EMBEDDINGS).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOR_EMBEDDINGS)
    outputs = []
    batch_size = 200
    with torch.no_grad():
        for idx in range(0, len(all_featnames), batch_size):
            tokens = tokenizer(text=all_featnames[idx:idx+batch_size], return_tensors='pt', max_length=128, truncation=True, padding=True).to(device)
            output = model(**tokens)
            outputs.append(output["pooler_output"])#.detach().to("cpu"))
            del output

    embeddings = torch.cat(outputs).cpu()
    kmeans = KMeans(n_clusters=50, random_state=5684, verbose=1)
    kmeans.fit(embeddings)
    clusters = kmeans.predict(embeddings)
    with open(CLUSTER_PATH, "wb") as f:
        np.save(f, clusters)

    return clusters
  

def ego_read_feats_cluster(egofeat_file, feat_file, featnames_file, featname2cluster):
    #Create a feature vector, which has as many dimensions as many clusters
    #And the value of each feature is the number of original features that node has
    #from that cluster
    x_ego = pd.read_csv(egofeat_file, sep=' ', header=None,dtype=np.float32)
    x_ego = torch.from_numpy(x_ego.values)

    x = pd.read_csv(feat_file, sep=' ', header=None, dtype=np.float32)
    x = torch.from_numpy(x.values)[:, 1:]

    x = torch.cat([x, x_ego], dim=0)

    with open(featnames_file, 'r') as f:
        featnames = f.read().split('\n')[:-1]
        featnames = [' '.join(x.split(' ')[1:]) for x in featnames]

    new_feats = []

    for feat_vector in x:
        vec = np.zeros(50, dtype=np.float32)
        for feat_index in feat_vector.nonzero().flatten():
            vec[featname2cluster[featnames[feat_index]]] += 1
        new_feats.append(vec)

    return torch.from_numpy(np.vstack(new_feats))
    
def ego_read_feats_gplus(egofeat_file, feat_file, featnames_file, feature2number, feature_types, max_count):
    #
    try:
        x_ego = pd.read_csv(egofeat_file, sep=' ', header=None,dtype=np.float32)
        x_ego = torch.from_numpy(x_ego.values)

        x = pd.read_csv(feat_file, sep=' ', header=None, dtype=np.float32)
        x = torch.from_numpy(x.values)[:, 1:]

        x = torch.cat([x, x_ego], dim=0)
    except Exception as e:
        #If there is an error during reading, that means that the files were bad
        #E.g. the egofeat or the feat files are empty
        return None

    with open(featnames_file, 'r') as f:
        featnames = f.read().split('\n')[:-1]
        featnames = [' '.join(x.split(' ')[1:]) for x in featnames]

    featname2index = {featname: i for i, featname in enumerate(featnames)}

    #Convert the features into a vector of dimension len(feature_types)*max_count
    #For each feature type, the first max_count will remain, and their indices
    #are kept in that feature_type
    new_feats = torch.zeros(x.size(0), len(feature_types)*max_count)
    for feat_index, feature_type in enumerate(feature_types):
        relevant_feats = []
        for f in featnames:
            if f.startswith(feature_type):
                relevant_feats.append(f)
        relevant_feat_indices = [featname2index[f] for f in relevant_feats]
        for i in range(x.size(0)):
            index = x[i, relevant_feat_indices].nonzero()
            if index.size(0) > 0:
                index = index[0:max_count]
                for idx_offset, idx in enumerate(index):
                    feature = relevant_feats[idx]
                    new_feats[i, feat_index*max_count+idx_offset] = feature2number[feat_index][feature]
            else:
                new_feats[i, feat_index] = -1

    return new_feats
    
    
def read_all_feats_and_fit_pca(files: str):
    #Fit PCA on all features in the dataset
    files = [
        x for x in files if x.split('.')[-1] in
        ['egofeat', 'feat', 'featnames']
    ]
    all_featnames = ego_read_all_featnames(files)
    all_featnames = {key: i for i, key in enumerate(all_featnames)}

    all_x = []

    for i in range(0, len(files), 3):
        try:
            egofeat_file = files[i + 0]
            feat_file = files[i + 1]
            featnames_file = files[i + 2]

            x_ego = pd.read_csv(egofeat_file, sep=' ', header=None,
                                dtype=np.float32)
            x_ego = torch.from_numpy(x_ego.values)

            x = pd.read_csv(feat_file, sep=' ', header=None, dtype=np.float32)
            x = torch.from_numpy(x.values)[:, 1:]

            x = torch.cat([x, x_ego], dim=0)
        except:
            continue

    # Reorder `x` according to `featnames` ordering.
    x_all = torch.zeros(x.size(0), len(all_featnames))
    with open(featnames_file, 'r') as f:
        featnames = f.read().split('\n')[:-1]
        featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
    indices = [all_featnames[featname] for featname in featnames]
    x_all[:, torch.tensor(indices)] = x
    x = x_all

    all_x.append(x)

    x_concat = torch.cat(all_x)
    pca = PCA(0.5, random_state=42)
    pca = pca.fit(x_concat)
    return pca
    
def read_ego(files: List[str], name: str, feature_mode:str) -> List[EgoData]:
    #Mostly from the pytorch geometric library, but modified slighty
    
    #Read all feature names from all graphs
    all_featnames = []
    files = [
        x for x in files if x.split('.')[-1] in
        ['circles', 'edges', 'egofeat', 'feat', 'featnames']
    ]
    for i in range(4, len(files), 5):
        featnames_file = files[i]
        with open(featnames_file, 'r') as f:
            featnames = f.read().split('\n')[:-1]
            featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
            all_featnames += featnames
    all_featnames = sorted(list(set(all_featnames)))
    all_featnames_names = all_featnames
    all_featnames = {key: i for i, key in enumerate(all_featnames)}

    if feature_mode=="cluster":
        clusters = cluster_featnames(all_featnames_names)
        featname2cluster = dict(zip(all_featnames_names, clusters.tolist()))

    if feature_mode=="pca":
        pca = read_all_feats_and_fit_pca(files)

    #Read all graphs (each has 5 files)
    data_list = []
    for i in tqdm(range(0, len(files), 5)):
        circles_file = files[i]
        edges_file = files[i + 1]
        egofeat_file = files[i + 2]
        feat_file = files[i + 3]
        featnames_file = files[i + 4]

        #Cases based on the feature_mode
        x = None
        #Basic means no additional preprocessing
        if feature_mode == None or feature_mode in ["basic", "pca"]:
            try:
                x_ego = pd.read_csv(egofeat_file, sep=' ', header=None,
                                dtype=np.float32)
            except Exception as e:
                print("BAD:", egofeat_file)
                continue
            x_ego = torch.from_numpy(x_ego.values)

            x = pd.read_csv(feat_file, sep=' ', header=None, dtype=np.float32)
            x = torch.from_numpy(x.values)[:, 1:]

            x = torch.cat([x, x_ego], dim=0)

            # Reorder `x` according to `featnames` ordering.
            x_all = torch.zeros(x.size(0), len(all_featnames))
            with open(featnames_file, 'r') as f:
                featnames = f.read().split('\n')[:-1]
                featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
            indices = [all_featnames[featname] for featname in featnames]
            x_all[:, torch.tensor(indices)] = x
            x = x_all

            if feature_mode=="pca":
                x = torch.from_numpy(pca.transform(x.numpy())).float()
        elif feature_mode=="diff":
            #Calculate differences on all feature types compared to the ego node
            feature_types = None
            if name == "facebook":
                feature_types = ["birthday;", "education;classes;id;", "education;concentration;id;",
                                 "education;degree;id;", "education;school;id;", "education;type;",
                                 "education;with;id", "education;year;id;", "first_name;",
                                 "gender;", "hometown;id;", "languages;id;",
                                 "last_name;", "locale;", "location;id;", "work;employer;id;",
                                 "work;end_date;", "work;location;id;", "work;position;id;",
                                 "work;start_date;", "work;with;id;"]
            elif name == "gplus":
                feature_types = ["gender:", "institution:", "job_title:", "last_name:", "place:", "university:"]
            elif name == "twitter":
                feature_types = ["#", "@"]
            x = read_ego_feats_as_diff(egofeat_file, feat_file, featnames_file, feature_types)
        elif feature_mode=="cluster":
            #Calculate features based on the clustering of the feature names
            x = ego_read_feats_cluster(egofeat_file, feat_file, featnames_file, featname2cluster)
        elif feature_mode=="gplus":
            feature_types = ["gender:", "institution:", "job_title:", "last_name:", "place:", "university:"]
            feature2number = []
            for feature_type in feature_types:
                relevant_feats = [feat for feat in all_featnames_names if feat.startswith(feature_type)]
                relevant_feats2number = {feat: i for i, feat in enumerate(relevant_feats)}
                feature2number.append(relevant_feats2number)
            x = ego_read_feats_gplus(egofeat_file, feat_file, featnames_file, feature2number, feature_types, 10)

        if x is None:
            continue

        idx = pd.read_csv(feat_file, sep=' ', header=None, dtype=str,
                          usecols=[0]).squeeze()

        #Reindex nodes
        idx_assoc = {}
        for i, j in enumerate(idx):
            idx_assoc[j] = i

        #The following part is fully from from pytorch geometric
        circles = []
        circles_batch = []
        with open(circles_file, 'r') as f:
            for i, circle in enumerate(f.read().split('\n')[:-1]):
                circle = [idx_assoc[c] for c in circle.split()[1:]]
                circles += circle
                circles_batch += [i] * len(circle)
        circle = torch.tensor(circles)
        circle_batch = torch.tensor(circles_batch)

        try:
            row = pd.read_csv(edges_file, sep=' ', header=None, dtype=str,
                              usecols=[0]).squeeze()
            col = pd.read_csv(edges_file, sep=' ', header=None, dtype=str,
                              usecols=[1]).squeeze()
        except:  # noqa
            continue

        row = torch.tensor([idx_assoc[i] for i in row])
        col = torch.tensor([idx_assoc[i] for i in col])

        N = max(int(row.max()), int(col.max())) + 2
        N = x.size(0) if x is not None else N

        row_ego = torch.full((N - 1, ), N - 1, dtype=torch.long)
        col_ego = torch.arange(N - 1)

        # Ego node should be connected to every other node.
        row = torch.cat([row, row_ego, col_ego], dim=0)
        col = torch.cat([col, col_ego, row_ego], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = coalesce(edge_index, num_nodes=N)

        data = EgoData(x=x, edge_index=edge_index, circle=circle,
                       circle_batch=circle_batch)

        data_list.append(data)

    return data_list
    
#From pytorch geometric
class SNAPEgoDataset(InMemoryDataset):
    r"""A variety of graph datasets collected from `SNAP at Stanford University
    <https://snap.stanford.edu/data>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://snap.stanford.edu/data'

    available_datasets = {
        'ego-facebook': ['facebook.tar.gz'],
        'ego-gplus': ['gplus.tar.gz'],
        'ego-twitter': ['twitter.tar.gz'],
    }

    def __init__(
        self,
        root: str,
        name: str,
        feature_mode,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        self.name = name.lower()
        self.feature_mode = feature_mode
        assert self.name in self.available_datasets.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return f"data_{self.feature_mode}.pt"

    def _download(self):
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        for name in self.available_datasets[self.name]:
            path = download_url(f'{self.url}/{name}', self.raw_dir)
            if name.endswith('.tar.gz'):
                extract_tar(path, self.raw_dir)
            elif name.endswith('.gz'):
                extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        raw_dir = self.raw_dir
        filenames = os.listdir(self.raw_dir)
        if len(filenames) == 1 and osp.isdir(osp.join(raw_dir, filenames[0])):
            raw_dir = osp.join(raw_dir, filenames[0])

        raw_files = sorted([osp.join(raw_dir, f) for f in os.listdir(raw_dir)])


        raw_graphs = sorted(set([x.split('.')[0] for x in raw_files]))
        if self.name[:4] == 'ego-':
          data_list = read_ego(raw_files, self.name[4:], self.feature_mode)
        else:
            raise NotImplementedError

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'SNAP-{self.name}({len(self)})'