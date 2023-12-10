import torch
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

class GCN(torch.nn.Module):
    def __init__(self, num_feat, hidden_dim, node_embedding_dim, hidden_dim_linear, num_layers, dropout):
        super().__init__()
        self.conv = nn.ModuleList(
        [GCNConv(num_feat, hidden_dim)] 
        + [GCNConv(hidden_dim, hidden_dim) for i in range(num_layers-2)]
        + [GCNConv(hidden_dim, node_embedding_dim)]
        )
        self.layer1 = Linear(node_embedding_dim*2, hidden_dim_linear)
        self.layer2 = Linear(hidden_dim_linear, 1)
        self.dropout = dropout

    def node_encoding(self, x, edge_index):
        for conv in self.conv:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def classifier(self, z, edge_index):
        edges1 = torch.index_select(z, 0, edge_index[0])
        edges2 = torch.index_select(z, 0, edge_index[1])
        data = torch.cat((edges1, edges2), 1)

        x = self.layer1(data)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)

        return F.sigmoid(x)

    def forward():
        pass

class Baseline():
    def fit_predict_baseline(train_data, val_data, test_data):
      data = []
      y = []
      for graph in train_data:
          edges1 = torch.index_select(graph.x, 0, graph.edge_label_index[0])
          edges2 = torch.index_select(graph.x, 0, graph.edge_label_index[1])
          data.append(torch.cat((edges1, edges2), 1))
          y.append(graph.edge_label)
      x = torch.cat(data, 0).to("cpu")
      y = torch.cat(y, 0).to("cpu")

      logreg = LogisticRegression(verbose = 1)
      logreg.fit(x, y)

      data = []
      y = []
      for graph in val_data:
          edges1 = torch.index_select(graph.x, 0, graph.edge_label_index[0])
          edges2 = torch.index_select(graph.x, 0, graph.edge_label_index[1])
          data.append(torch.cat((edges1, edges2), 1))
          y.append(graph.edge_label)
      x_val = torch.cat(data, 0).to("cpu")
      y_val = torch.cat(y, 0).to("cpu")


      y_all = []
      y_probas_all = []
      for graph in test_data:
        X, y = create_x_y(graph)
        X = X.cpu()
        y = y.cpu()
        y_probas = logreg.predict_proba(X)[:,1]

        y_all.append(y)
        y_probas_all.append(y_probas)

      y_all = torch.cat(y_all).numpy()
      y_probas_all_baseline = np.hstack(y_probas_all)

      return y_all, y_probas_all_baseline

