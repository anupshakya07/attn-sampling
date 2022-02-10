# Scaling up using Attention

#### In this project, we use the famous attention mechanism to discover symmetries in the dataset and use it to our advantage to sample a small but highly informative set of training samples to efficiently train an ML model with high accuracy. More specifically, we identify the most important regions in the token in the output sequence and prune the less important ones. We use non-parametric clustering to get the most optimal set of clusters, which we train iteratively by conditioning it on a similarity metric. We sample from the clusters and train an attention model.



# Acknowledgements

The implementations for DP-means and Transformers were taken from the following source:
- https://github.com/DrSkippy/Python-DP-Means-Clustering
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
