# Deep learning homework project - Friend recommendation with graph neural networks 

## Team name

SzociAI

## Team members' names and Neptun codes

- Gergály Anna WWPD4V
- Mészáros Péter RBNJB7

## Project description

The goal of this project is to develop a personalized friend recommendation system by using Graph Neural Networks (GNNs). We analyze data from Facebook, Google+, or Twitter to suggest meaningful connections based on user profiles and interactions.

## Functions of the files in the repository

- **Dockerfile**  
The initial version of the dockerfile for containerization
- **Friend_recommendation.ipynb**  
The main ipython notebook containing the code for data acquisition, analysis and preparation. (The notebook was mainly used and tested on Google Colab.
- **requirements.txt**  
The list of necessary python packages
- **twitter_featurenames_clusters**  
Saved artifact of a word clustering used for data preparation. Contains a cluster index for each feature name in the Twitter dataset. The project can be run without it present but it greatly speeds up the process when present, because without it the clustering has to be done from scratch.

## Related works
- [Learning to Discover Social Circles in Ego Networks](http://i.stanford.edu/~julian/pdfs/nips2012.pdf)
- [Pytorch Geometric](https://www.example.com](https://github.com/pyg-team/pytorch_geometric)https://github.com/pyg-team/pytorch_geometric)
- [NetworkX](https://www.example.com](https://github.com/networkx/networkx)https://github.com/networkx/networkx)

## How to run

In the "main" directory, build the container from the docker file.

```
docker build . -t dl:friend_recommendation
```
To run the container:
```
docker run --rm -it dl:friend_recommendation bash
```

In the container run jupyter-lab to open the notebook, jupyter lab can be accessed from the host on port 8887.

```
jupyter-lab --ip 0.0.0.0 --port 8887 --no-browser --allow-root
```
