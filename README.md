# Scalable and Equitable Math Problem Solving Strategy Prediction in Big Educational Data

<!--- #### In this project, we use the famous attention mechanism to discover symmetries in the dataset and use it to our advantage to sample a small but highly informative set of training samples to efficiently train an ML model with high accuracy. More specifically, we identify the most important regions/tokens in the output sequence and prune the less important ones. We use non-parametric clustering to get the most optimal set of clusters, which we train iteratively by conditioning it on a similarity metric. We sample from the clusters and train an attention model. The attention model in turn gives the important regions for every strategy in the dataset. We learn new embeddings based on this and train the strategy prediction model. Based on the mastery, we predict the strategy and based on the strategies, we modify the mastery model. -->
# Abstract
Understanding a student's problem-solving strategy can have a significant impact on effective math learning using Intelligent Tutoring Systems (ITSs) and Adaptive Instructional Systems (AISs). For instance, the ITS/AIS can better personalize itself to correct specific misconceptions that are indicated by incorrect strategies, specific problems can be designed to improve strategies and frustration can be minimized by adapting to a student's natural way of thinking rather than trying to fit a standard strategy for all. While it may be possible for human experts to identify strategies manually in classroom settings with sufficient student interaction, it is not possible to scale this up to big data. Therefore, we leverage advances in Machine Learning and AI methods to perform scalable strategy prediction that is also fair to students at all skill levels. Specifically, we develop an embedding called MVec where we learn a representation based on the mastery of students. We then cluster these embeddings with a non-parametric clustering method where each cluster contains instances that have approximately symmetrical strategies. The strategy prediction model is trained on instances sampled from these clusters ensuring that we train the model over diverse strategies. Using real world large-scale student interaction datasets from MATHia, we show that our approach can scale up to achieve high accuracy by training on a small sample of a large dataset and also has predictive equality, i.e., it can predict strategies equally well for learners at diverse skill levels.

# Requirements
- Python 3.8
- gensim 3.8.3
- Tensorflow 2.6.0
- keras 2.6.0
- torch 1.13.1
- pandas 1.3.4
- tqdm
- scikit-learn 1.6.1

# How to use this code

This project has four parts. First, we preprocess the dataset followed by finding the optimal clusters. Then, we prune relations in the sequences and finally train the strategy prediction model. The required parameters and commands to run the scripts in order is given below:

### 1. Preprocess the Dataset and create vector embeddings for objects in the dataset.

```
preprocess.py -dataset_path "dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt" -embedding_model_output "embedding-models/gensim-bridge-to-algebra-embedding-model" -construct_vocab=True
```
#### Configurable parameters for this script:
- dataset_path - the path to the dataset
- embedding_model_output - the path to the output file where the embedding model will be saved
- construct_vocab - boolean flag that determines the construction of new vocab for the dataset

### 2. Run the DP-means clustering algorithm, sample from the clusters, train the KC-CFA attention model and converge to optimal set of clusters.

```
dp_means_optimizer.py -dataset "dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt" -embedding_model "embedding-models/gensim-bridge-to-algebra-embedding-model" -data_path "output/tokenized_data_BOS_EOS.pkl" -use_gpu=True -sample_new_data=True -recluster=True
```
#### Configurable parameters for dp_means_optimizer.py:
- dataset - the path to the dataset file
- embedding_model - the path to the vector embedding model for the dataset
- data_path - the path to the preprocessed output file generated from process.py (defaults to "output/tokenized_data_BOS_EOS.pkl")
- epoch - the number of epochs to train the KC-CFA attention model
- batch_size - the size of the batch to be used in each forward pass through the attention model
- use_gpu - boolean flag to control the use of cuda-enabled GPU device (defaults to True, but can be run on CPU with reduced performance)
- save_mode - the parameter to control saving the attention models at the end of each epoch. Options are "best" and "all" (defaults to "best")
- warmup_steps - the number of steps to use for warming up the attention model. the learning rate doesn't change when the model is warming up (defaults to 4000)
- output_dir - the directory to save the output attention models in
- sample_new_data - boolean flag to control sampling new data in each iteration
- recluster - boolean flag to control re-running the DP-means clustering in each iteration

### 3. Prune the weak relations and relearn the vector embeddings based on attention scores
```
prune_relations.py -dataset "dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt" -data_path "output/tokenized_data_BOS_EOS.pkl" -embedding_model "embedding-models/gensim-bridge-to-algebra-embedding-model" -use_gpu=True -clip_kcs=True -prune_embeddings=True -train_embeddings=True
```

#### Configurable parameters for prune_relations.py:
- dataset - the path to the dataset file
- embedding_model - the path to the vector embedding model for the dataset
- data_path - the path to the preprocessed output file generated from process.py (defaults to "output/tokenized_data_BOS_EOS.pkl")
- use_gpu - boolean flag to control the use of cuda-enabled GPU device (defaults to True, but can be run on CPU with reduced performance)
- clip_kcs - boolean flag to control pruning weak KC relations in the dataset
- train_embeddings - boolean flag to control learning new embeddings with pruned KCs
- prune_embeddings - boolean flag to control obtaining new set of vector embeddings by pruning the original embeddings
- mastery_model - the path to the saved mastery model (checkpoint file) from dp_means_optimizer.py (defaults to "output/models/mastery_model/model.chkpt")

### 4. Learn the strategy prediction model

```
learn_prediction_model.py -dataset "dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt" -test_file_path "dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_test.txt" -embedding_model "embedding-models/pruned-embedding-model-75-percent-retained" -epoch 10 -batch_size 10 -num_students 10 -num_problems 100
```
#### Configurable Parameters for learn_prediction_model.py:
- dataset - the path to the dataset file
- test_file_path - the path to the test data
- embedding_model - the path to the attention-pruned vector embedding model for the dataset
- epoch - the number of epochs to train the LSTM-NS model
- batch_size - the size of the batch to be used in each forward pass through the LSTM-NS model
- num_students - the number of student clusters to sample in each forward pass
- num_problems - the number of problem clusters to sample in each forward pass

# Acknowledgements

The implementations for DP-means and Transformers were taken from the following source:
- https://github.com/DrSkippy/Python-DP-Means-Clustering
- https://github.com/jadore801120/attention-is-all-you-need-pytorch

# Citing

Please consider citing.

@inproceedings{shakya2023, 
               author = {Anup Shakya and Vasile Rus and Deepak Venugopal}, 
               title = {Scalable and Equitable Math Problem Solving Strategy Prediction in Big Educational Data},
               booktitle = {Sixteenth International Conference on Educational Data Mining 2023}, 
               year = {2023}
}
