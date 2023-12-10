import numpy as np
from urllib.request import urlretrieve
import gradio as gr
from torch_geometric.datasets.snap_dataset import EgoData
from torch_geometric.utils import coalesce,to_networkx
import json
import torch
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
import os
from models import GCN


def read_graph_from_json(file):
    #Reads a graph from a json file
    with open(file, "rt") as f:
        nodes = json.load(f)["nodes"]
    ids = [node["id"] for node in nodes]
    id2index = {id:index for index, id in enumerate(ids)}

    if "name" in nodes[0]:
        labeldict = {id2index[node["id"]]:node["name"] for node in nodes}
    else:
        labeldict = {id2index[node["id"]]:"" for node in nodes}

    all_edges = []
    for node in nodes:
        node_index = id2index[node["id"]]
        all_edges.extend([ [node_index, id2index[neighbor]] for neighbor in node["neighbors"] ])

    N = len(id2index)

    edge_index = torch.tensor(all_edges).T
    edge_index = coalesce(edge_index, num_nodes=N)

    x = torch.tensor([node["x"] for node in nodes], dtype=torch.float32)

    return EgoData(x=x, edge_index=edge_index), id2index, labeldict
  
 
def get_friends(model, graph, person_index):
    #For a given person, predicts top 3 new friends from the nodes which are not friends of the person
    current_neighbors = set(graph.edge_index.T[(graph.edge_index[0] == person_index).nonzero()].flatten(end_dim=1).T[1].tolist())
    current_neighbors.add(person_index)
    neighbor_candidates = torch.tensor([x for x in range(graph.num_nodes) if x not in current_neighbors])
    not_neighbor_edges = torch.stack((torch.full(neighbor_candidates.size(), person_index), neighbor_candidates), dim=0)

    model.eval()
    with torch.no_grad():
        z = model.node_encoding(graph.x, graph.edge_index)
        out = model.classifier(z, not_neighbor_edges).view(-1)

    return neighbor_candidates[torch.topk(out, 3)[1]].tolist()
  
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    num_layers = int(len([x for x in checkpoint["model_state_dict"].keys() if x.startswith("conv")])/2)
    num_feat = checkpoint["model_state_dict"]["conv.0.lin.weight"].shape[1]
    hidden_dim = checkpoint["model_state_dict"][f"conv.{num_layers-1}.lin.weight"].shape[1]
    node_embedding_dim = checkpoint["model_state_dict"][f"conv.{num_layers-1}.lin.weight"].shape[0]
    hidden_dim_linear = checkpoint["model_state_dict"]["layer2.weight"].shape[1]
    
    model = GCN(num_feat, hidden_dim, node_embedding_dim, hidden_dim_linear, num_layers, 0.5)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model
    
  
def read_graph(model_path, file, person):
    model = load_model(model_path)
   
    if file.endswith(".json"):
        graph, id2index, labeldict = read_graph_from_json(file)

    index2id = {index:id for id, index in id2index.items()}

    highlight = []

    if type(person) == str:
        label2id = {label:id for id, label in labeldict}
        highlight = [id2index[label2id[person]]]
    elif type(person) == int or type(person) == float:
        highlight = [id2index[person]]

    person_index = highlight[0]

    friends = get_friends(model, graph, person_index)

    plt.figure()
    G = to_networkx(graph, to_undirected=False)
    pos = nx.spring_layout(G, iterations=50, k=0.4, seed=175)
    nx.draw(G, pos, alpha = 0.8, node_color='gray', edge_color="gray", with_labels=True, font_color="black", font_size=14, labels=labeldict)
    nx.draw_networkx_nodes(G, pos, nodelist=highlight, node_color="orange")
    nx.draw_networkx_nodes(G, pos, nodelist=friends, node_color="lightgreen")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image.load()
    buf.close()

    friends = [f"({labeldict[friend]},{index2id[friend]})" for friend in friends]
    friends = ", ".join(friends)

    return image, friends
  
def main():

    models = [f for f in os.listdir("models")]
    model_paths = [os.path.join("models", f, "best.pt") for f in models]
        
    dropdown = gr.Dropdown(
            choices = models, value=model_paths, label="Model"
        )

    gr.Interface(fn=read_graph,
             inputs=[dropdown, "file", "number"],
             outputs=["image", "text"],
             title="Friend recommendation").launch(share=True)
    
if __name__=="__main__":
    main()