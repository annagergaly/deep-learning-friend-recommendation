# Deep learning homework project - Friend recommendation with graph neural networks 

## Team name

SzociAI

## Team members' names and Neptun codes

- Gergály Anna WWPD4V
- Mészáros Péter RBNJB7

## Project description

The goal of this project is to develop a personalized friend recommendation system by using Graph Neural Networks (GNNs). We analyze data from Facebook, Google+, or Twitter to suggest meaningful connections based on user profiles and interactions.

## Functions of the files in the repository

- **models**  
Model checkpoints saved for running inference on gradio
- **samples**  
Sample json files as examples for gradio
- **Dockerfile**  
The initial version of the dockerfile for containerization
- **Friend_recommendation.ipynb**  
The main ipython notebook containing the code for data acquisition, analysis and preparation, using our other code split into separate python files. (The notebook was mainly used and tested on Google Colab.)
- **dataset_utils.py**  
Utility python code concerning dataset loading. It contains the SNAPEgoDataset class, which handles downloading and preprocessing.
- **gradio_utils.py**  
Utility python code enabling gradio integration for user interface.
- **model_utils.py**  
Utility functions for training, validating and running inference on models.
- **models.py**  
File containing the code for our model and the baseline.
- **requirements.txt**  
The list of necessary python packages.
- **twitter_featurenames_clusters**  
Saved artifact of a word clustering used for data preparation. Contains a cluster index for each feature name in the Twitter dataset. The project can be run without it present but it greatly speeds up the process when present, because without it the clustering has to be done from scratch.

## Related works
- [Learning to Discover Social Circles in Ego Networks](http://i.stanford.edu/~julian/pdfs/nips2012.pdf)
- [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [NetworkX](https://github.com/networkx/networkx)
- [Graph: Train, valid, and test dataset split for link prediction](https://zqfang.github.io/2021-08-12-graph-linkpredict/)
- [Graph Neural Networks with PyG on Node Classification, Link Prediction, and Anomaly Detection](https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275)
- [Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking](https://arxiv.org/abs/2306.10453)
- [A comprehensive survey of edge prediction in social networks: Techniques, parameters and challenges](https://www.sciencedirect.com/science/article/pii/S0957417419300466)
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

## How to run

In the "main" directory, build the container from the docker file.

```
docker build . -t dl:friend_recommendation
```
To run the container:
```
docker run --rm -it -p 7860:7860 -p 8887:8887 dl:friend_recommendation bash
```
In the container run jupyter-lab to open the notebook, jupyter lab can be accessed from the host on port 8887.

```
jupyter-lab --ip 0.0.0.0 --port 8887 --no-browser --allow-root
```

Training and evaluation is done in the Friend_recommendation.ipynb notebook. For the entire pipeline run the notebook all the way through. 

For training, to train the deep learning models use the train_model function, to train the baseline solution run train_baseline_all_graphs.

For evaluation, run the Evaluation section of the notebook.

To run the user interface:
```
python gradio_utils.py
```
This will start a Gradio server, and it will be accessible on the host machine on http://127.0.0.1:7860/.
