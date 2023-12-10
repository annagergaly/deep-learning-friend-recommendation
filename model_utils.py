import torch
import torch_geometric
import numpy as np
import wandb
import os

def train_epoch(model, train_data, optimizer, loss_fct):
    loss_acc = 0
    acc_nom = 0
    acc_den = 0
    for graph in train_data:
        optimizer.zero_grad()

        z = model.node_encoding(graph.x, graph.edge_index)
        out = model.classifier(z, graph.edge_label_index).view(-1)
        loss = loss_fct(out, graph.edge_label)
        loss.backward()

        optimizer.step()

        loss_acc += loss

        acc_nom += (out.round() == graph.edge_label).sum()
        acc_den += len(graph.edge_label)

    acc = acc_nom/acc_den
    loss_acc = loss_acc/len(train_data)

    return {"train/train_loss": loss_acc, "train/train_accuracy": acc}
	

def validation(model, val_data, loss_fct):
    loss_acc = 0
    acc_nom = 0
    acc_den = 0
    with torch.no_grad():
        for graph in val_data:
            z = model.node_encoding(graph.x, graph.edge_index)
            out = model.classifier(z, graph.edge_label_index).view(-1)
            loss_acc += loss_fct(out, graph.edge_label)

            acc_nom += (out.round() == graph.edge_label).sum()
            acc_den += len(graph.edge_label)
    loss_acc = loss_acc/len(val_data)
    acc = acc_nom/acc_den
    return {"val/val_loss": loss_acc, "val/val_accuracy": acc}
	
def train_model(model, train_data, val_data, epochs, epoch_counter, patience, model_name, log):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fct = torch.nn.BCEWithLogitsLoss()

    early_stopping_counter = 0
    delta = 0.001
    min_val_loss = float('inf')

    for epoch in range(epochs):
        metrics = train_epoch(model, train_data, optimizer, loss_fct)
        metrics.update(validation(model, val_data, loss_fct))

        if metrics["val/val_loss"] < min_val_loss:
            min_val_loss = metrics["val/val_loss"]
            early_stopping_counter = 0
            os.makedirs(os.path.join("models", model_name), exist_ok=True)
            torch.save({
                "epoch":epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": metrics["train/train_loss"]
            }, os.path.join("models", model_name, "best.pt"))

        if metrics["val/val_loss"] > min_val_loss + delta:
            early_stopping_counter += 1
            if early_stopping_counter > patience:
                break
                
        metrics["epoch"] = epoch
        if log:
            wandb.log(metrics)
            wandb.save(os.path.join("models", model_name, "best.pt"))

        if epoch % epoch_counter == 0:
            print(f'Epoch: {epoch:02d}, Train loss: {metrics["train/train_loss"]:.4f}, Train acc {metrics["train/train_accuracy"]:.2%}, '
            f'Val loss: {metrics["val/val_loss"]:.4f}, Val acc {metrics["val/val_accuracy"]:.2%}')

    print(f'Epoch: {epoch:02d}, Train loss: {metrics["train/train_loss"]:.4f}, Train acc {metrics["train/train_accuracy"]:.2%}, '
            f'Val loss: {metrics["val/val_loss"]:.4f}, Val acc {metrics["val/val_accuracy"]:.2%}')
			
def predict(model, graph):
    with torch.no_grad():
        z = model.node_encoding(graph.x, graph.edge_index)
        out = model.classifier(z, graph.edge_label_index).view(-1)
    return out
    
def preds_from_model(model, test_data):
  y_preds = []
  for graph in test_data:
    y_preds.append(predict(model, graph))

  y_preds = torch.cat(y_preds).cpu().numpy()

  return y_preds
	
def test(model, test_data):
    model.eval()
    out = []
    for graph in test_data:
        out.append(predict(model, graph))
    return torch.cat(out)