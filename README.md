# TSCI
TSCI (Trajectory-based Social Circle Inference) <br>
This is the implementation of the deep learning based TSCI model: DeepTSCI.

# Environment
* python >= 2.7
* Tensorflow 1.0 or ++

# Trajectory splitting and POI embedding
* To capture mobility patterns of trajectories, We first split each trajectory into sub-trajectories based on the fixed time interval (see the paper for the detailed explanation). 
* Next, we embed POIs into low-dimensional vectors using the common method: wordv2vec.

# Train
# Evaluation
# Dataset
We use four different Location-based Social Network data as follows. 
* Gowalla:<http://snap.stanford.edu/data/loc-gowalla.html>
* Brightkite:<http://snap.stanford.edu/data/loc-brightkite.html>
* Foursquare(New York,Tokyo):<https://sites.google.com/site/yangdingqi/home/foursquare-dataset>
* (remark) Please do not use these datasets for commercial purpose. For academic uses, please cite the paper. We use the same method as in [41] to construct the social networks. Thanks for their help.(see the reference [41] in the paper).

# Performance

# Usage
*To run DeepTSCI, python GW_LSTM.py/GW_BLSTM.py. The outcome including model and results will be in the folder of out_data.
*To run VAE-based DeepTSCI, run the following commands (pre-training and training):
*  python GW_VAE.py 
*  python GW_VAE_S.py

# Reference
Hope such an implementation could help you on your projects. Any comments and feedback are appreciated. 
