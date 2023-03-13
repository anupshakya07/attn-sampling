import pickle
import random

import gensim
import torch
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.attention_model import MathiaDataset, train, collate_data
from dpmeans.cluster import dpmeans
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from transformer.Models import PositionalEncoding
from tqdm import tqdm

ITER_CLUSTER = 10
MIN_SEQ_LENGTH = 1
MAX_SEQ_LENGTH = 150
BOS_TOKEN = 3  # Beginning of sentence token
EOS_TOKEN = 4  # End of sentence token


def iterate(std_vectors, prob_vectors, embedding_model, kc_index_map, device, LAMBDA, opt):
    std_cluster, prob_cluster = cluster(std_vectors, prob_vectors, LAMBDA)
    pickle.dump(std_cluster, open("output/student_cluster.pkl", "wb"))
    pickle.dump(prob_cluster, open("output/problem_cluster.pkl", "wb"))

    std_samples = generate_samples(std_cluster, 50)  # tested with 10 for the first iteration
    students = words_from_embeddings(std_samples, embedding_model)
    all_std_list = []
    for key, std_list in students.items():
        all_std_list.extend(std_list)

    print("Total number of students sampled = ", len(all_std_list))

    prob_samples = generate_samples(prob_cluster, 100)  # tested with 20 for the first iteration
    problems = words_from_embeddings(prob_samples, embedding_model)
    all_prob_list = []
    for key, prob_list in problems.items():
        all_prob_list.extend(prob_list)
    print("Total number of problems sampled = ", len(all_prob_list))

    data_iterator = pd.read_csv(opt.dataset, sep="\t", header=0, iterator=True,
                                chunksize=1000000,
                                usecols=["Anon Student Id", "Problem Name", "Problem Hierarchy",
                                         "KC(SubSkills)", "Correct First Attempt"])

    sampled_data = []
    sampled_data_with_clusters = dict()
    input_vector_sequence = []
    output_raw_sequence = []
    max_length = 0

    for data_chunk in data_iterator:
        for stdnt, std_group in data_chunk.groupby("Anon Student Id"):
            if stdnt in all_std_list and stdnt in embedding_model.wv.vocab:
                std_cluster_id = determine_cluster_id(stdnt, students)

                for prob_hierarchy, prob_hierarchy_groups in std_group.groupby("Problem Hierarchy"):
                    if prob_hierarchy in embedding_model.wv.vocab:

                        for prblm, prob_group in prob_hierarchy_groups.groupby("Problem Name"):
                            if prblm in all_prob_list and prblm in embedding_model.wv.vocab:
                                prob_cluster_id = determine_cluster_id(prblm, problems)
                                cluster_key = ",".join([str(std_cluster_id), str(prob_cluster_id)])
                                cluster_samples = []
                                input_inst = [embedding_model.wv[stdnt],
                                              embedding_model.wv[prob_hierarchy],
                                              embedding_model.wv[prblm]]
                                output_inst = []

                                subskills = prob_group['KC(SubSkills)']
                                cfas = prob_group['Correct First Attempt']
                                kcs_prob = [BOS_TOKEN]
                                cfa_list = [BOS_TOKEN]

                                for kc_row, cfa_row in zip(subskills, cfas):
                                    if str(kc_row) != "nan":
                                        correct_first_attempt = cfa_row
                                        splitted_kcs = kc_row.split("~~")
                                        for kc in splitted_kcs:
                                            if kc in embedding_model.wv.vocab:
                                                kcs_prob.append(kc_index_map[kc])
                                                output_inst.append(kc)
                                                cfa_list.append(correct_first_attempt)

                                if len(kcs_prob) > 1:
                                    cfa_list.append(EOS_TOKEN)
                                    kcs_prob.append(EOS_TOKEN)
                                    input_vector_sequence.append(input_inst)
                                    output_raw_sequence.append(output_inst)
                                    if len(cfa_list) > max_length:
                                        max_length = len(cfa_list)
                                    # sampled_data.append({"src": kcs_prob, "target": cfa_list})
                                    cluster_samples.append({"src": kcs_prob, "target": cfa_list})
                                    sampled_data.extend(cluster_samples)
                                    # sampled_data_with_clusters.update({cluster_key: cluster_samples})
                                    if cluster_key in sampled_data_with_clusters:
                                        sampled_data_with_clusters[cluster_key].append(
                                            {"src": kcs_prob, "target": cfa_list})
                                    else:
                                        sampled_data_with_clusters[cluster_key] = cluster_samples

    print("X Input List Length -> ", len(sampled_data))
    train_data, val_data = train_test_split(sampled_data, test_size=0.1, shuffle=True, random_state=40)
    data_dump = {
        "max_length": max_length,
        "train": train_data,
        "val": val_data,
        "kc_index_map": kc_index_map,
        "lambda": LAMBDA
    }
    data_dump_file_name = "output/sampled_data_lambda_val_" + str(lambda_val) + ".pkl"
    cluster_data_file_name = "output/sampled_data_with_clusters_lambda_val_" + str(lambda_val) + ".pkl"
    pickle.dump(data_dump, open(data_dump_file_name, "wb"))
    pickle.dump(sampled_data_with_clusters, open(cluster_data_file_name, "wb"))

    train_dataset = MathiaDataset(data_path=data_dump_file_name, embedding_model_path=opt.embedding_model,
                                  dataset_type="train")
    train_dataset.set_max_seq_length(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_data,
                                  drop_last=True)

    validation_dataset = MathiaDataset(data_path=data_dump_file_name, embedding_model_path=opt.embedding_model,
                                       dataset_type="val")
    validation_dataset.set_max_seq_length(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
    validation_dataloader = DataLoader(validation_dataset, batch_size=opt.batch_size, shuffle=False,
                                       collate_fn=collate_data, drop_last=True)

    n_src_vocab = len(kc_index_map) + 5

    transformer = Transformer(n_src_vocab=n_src_vocab,
                              n_trg_vocab=2 + 3,
                              src_pad_idx=2,
                              trg_pad_idx=2,
                              trg_emb_prj_weight_sharing=False,
                              emb_src_trg_weight_sharing=False,
                              n_position=300)

    optimizer = ScheduledOptim(optimizer=torch.optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
                               lr_mul=2.0, d_model=512, n_warmup_steps=opt.warmup_steps)

    train(transformer, train_dataloader, validation_dataloader, optimizer, device, opt)

    max_length = MAX_SEQ_LENGTH if MAX_SEQ_LENGTH >= train_dataset.max_length else train_dataset.max_length
    positional_encoding = PositionalEncoding(300, max_length).to(device)

    data_with_clusters = pickle.load(open(cluster_data_file_name, "rb"))
    overall_clustering_scores = []
    overall_clustering_attention_scores = []

    processed_attention_scores = pickle.load(open("output/processed_attention_scores.pkl", "rb"))
    cluster_similarity_scores = dict()
    cluster_attention_scores = dict()

    for cluster_key, cluster_samples in tqdm(data_with_clusters.items(), desc="Clusters Loop"):
        cluster_sim = []
        cluster_attn_sim = []
        for i in tqdm(range(0, len(cluster_samples) - 1), desc="Internal Loop"):

            seq1_with_bos_eos = cluster_samples[i]['src']
            seq2_with_bos_eos = cluster_samples[i + 1]['src']

            seq1_str = ",".join([str(ist) for ist in seq1_with_bos_eos])
            seq2_str = ",".join([str(ist) for ist in seq2_with_bos_eos])

            seq1_attn = None
            seq2_attn = None

            if seq1_str in processed_attention_scores and seq2_str in processed_attention_scores:
                seq1_attn = processed_attention_scores[seq1_str]
                seq2_attn = processed_attention_scores[seq2_str]

                attn_dot_product = get_dot_product(seq1_attn, seq2_attn)
                seq1_attn_norm = np.linalg.norm(seq1_attn)
                seq2_attn_norm = np.linalg.norm(seq2_attn)
                attn_similarity = np.divide(attn_dot_product, seq1_attn_norm * seq2_attn_norm)
                cluster_attn_sim.append(attn_similarity)
                # print("attention similarity = ", attn_similarity)

            seq1 = seq1_with_bos_eos[1:-1]
            seq2 = seq2_with_bos_eos[1:-1]
            seq1_pos = calculate_positional_embedding(seq1, positional_encoding, embedding_model, index_kc_map, device)
            seq2_pos = calculate_positional_embedding(seq2, positional_encoding, embedding_model, index_kc_map, device)
            sim = get_alignment_similarity(seq1_pos, seq2_pos)
            cluster_sim.append(sim)

        if cluster_sim:
            mean = np.array(cluster_sim).mean()
            overall_clustering_scores.append(mean)
            cluster_similarity_scores[cluster_key] = mean
            # print("Average Cluster similarity for cluster ", cluster_key, " = ", np.array(cluster_sim).mean())

        if cluster_attn_sim:
            mean = np.array(cluster_attn_sim).mean()
            overall_clustering_attention_scores.append(mean)
            cluster_attention_scores[cluster_key] = mean
    print("Clustering Score Mean = ", np.array(overall_clustering_scores).mean())
    print("Clustering Similarity Score Mean = ", np.array(overall_clustering_attention_scores).mean())
    pickle.dump(cluster_similarity_scores, file=open("output/cluster_similarity_scores.pkl", "wb"))
    pickle.dump(cluster_attention_scores, file=open("output/cluster_attention_scores.pkl", "wb"))
    return np.array(overall_clustering_scores).mean(), np.array(overall_clustering_attention_scores).mean()


def cluster(std_vectors, prob_vectors, val_lambda):
    std_dp_means = dpmeans(std_vectors, _lam=val_lambda)
    prb_dp_means = dpmeans(prob_vectors[:], _lam=val_lambda)
    s_err, s_err_X = std_dp_means.run(ITER_CLUSTER)
    print("Student clustering error -> ", s_err)
    p_err, p_err_X = prb_dp_means.run(ITER_CLUSTER)
    print("Problem clustering error -> ", p_err)
    return std_dp_means, prb_dp_means


def words_from_embeddings(embeddings_map, embedding_model):
    words_map = dict()
    for i, emb_list in embeddings_map.items():
        word_list = []
        for emb in emb_list:
            word_list.append(embedding_model.wv.most_similar(positive=[emb], topn=1)[0][0])
        words_map.update({i: word_list})
    return words_map


def determine_cluster_id(item, item_dict):
    for cluster_id, item_list in item_dict.items():
        for inst in item_list:
            if inst == item:
                std_cluster_id = cluster_id
                return std_cluster_id


def generate_samples(cluster, n_samples):
    num_clusters = cluster.k
    final_sample = dict()
    for i in range(num_clusters):
        cluster_indices = [idx for idx, x in enumerate(cluster.dataClusterId) if x == i]
        if len(cluster_indices) < n_samples:
            n = len(cluster_indices)
        else:
            n = n_samples
        samples = random.sample(cluster_indices, n)
        sample_vectors = [cluster.X[idx] for idx in samples]
        final_sample.update({i: sample_vectors})
    return final_sample


def calculate_positional_embedding(instance, positional_encoding, emb_model, index_map, device):
    instance = [index_map[index] for index in instance]
    instance_wv = []
    for kc in instance:
        if kc in emb_model.wv.vocab:
            instance_wv.append(emb_model.wv[kc])
    # instance = [emb_model.wv[kc] for kc in instance]
    inst_tensor = torch.tensor(instance_wv, device=device).unsqueeze(0)
    positional_embedding = positional_encoding(inst_tensor)
    return positional_embedding


def get_dot_product(arr1, arr2):
    arr1_len = len(arr1)
    arr2_len = len(arr2)

    pad_length = (arr1_len - arr2_len) if arr1_len > arr2_len else (arr2_len - arr1_len)
    pad = [0] * pad_length
    arr2.extend(pad) if arr1_len > arr2_len else arr1.extend(pad)
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    dot_product = np.dot(arr1, arr2)
    return dot_product


def get_alignment_similarity(tensor1, tensor2):
    product = torch.matmul(tensor1, tensor2.transpose(1, 2))
    # dropout = nn.Dropout(0.1)
    product_softmaxed = F.softmax(product, dim=-1)
    alignment_matrix = dict()
    for i in range(tensor1.shape[1]):
        row_np_arr = product_softmaxed[0][i].cpu().detach().numpy()
        row_max = row_np_arr.max().item()
        row_mean = row_np_arr.mean().item()

        row_max_indices = np.where(row_np_arr == row_max)[0]
        row_max_index = row_max_indices[0].item() if len(row_max_indices) > 1 else row_max_indices.item()
        alignment_matrix[i] = [row_max_index, row_max, row_mean]
    alignment_matrix

    positive = 0
    negative = 0
    prev_match = -1
    for i in range(len(alignment_matrix)):
        penalty = True
        if alignment_matrix[i][0] > prev_match:
            penalty = False
        prev_match = alignment_matrix[i][0]
        weight = alignment_matrix[i][1] - alignment_matrix[i][2]
        if penalty:
            negative = negative + weight
        else:
            positive = positive + weight

    similarity = (positive - negative) / len(alignment_matrix)

    return similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str,
                        default="dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt")
    parser.add_argument('-embedding_model', type=str,
                        default="embedding-models/gensim-bridge-to-algebra-embedding-model")
    parser.add_argument('-data_path', type=str, default="output/tokenized_data_BOS_EOS.pkl")

    parser.add_argument('-epoch', type=int, default=10)

    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-use_gpu', type=bool, default=False)

    parser.add_argument('-batch_size', type=int, default=10)

    parser.add_argument('-save_mode', type=str, default='best')

    parser.add_argument('-warmup_steps', type=int, default=4000)

    parser.add_argument('-output_dir', type=str, default='output/models/mastery_model')

    parser.add_argument('-sample_new_data', type=bool, default=False)

    parser.add_argument('-recluster', type=bool, default=False)

    opt = parser.parse_args()

    s_vectors = pickle.load(open("input/unique_student_embeddings.pkl", "rb"))
    p_vectors = pickle.load(open("input/unique_problems_embeddings.pkl", "rb"))
    wv_model = gensim.models.Word2Vec.load(opt.embedding_model)

    all_data = pickle.load(open(opt.data_path, "rb"))
    index_kc_map = all_data["index_kc_map"]
    kc_index_map = all_data["kc_index_map"]
    hardware_device = torch.device('cuda' if opt.use_gpu else 'cpu')

    lambda_val = 9
    reiterate = True
    previous_similarity_score = 0
    while reiterate:
        positional_similarity, attention_similarity = iterate(s_vectors, p_vectors, wv_model, kc_index_map,
                                                              hardware_device, lambda_val, opt)
        similarity_score = (positional_similarity + attention_similarity) / 2

        print("Similarity Positional -> ", positional_similarity)
        print("Similarity Attention -> ", attention_similarity)
        print("Similarity Score Overall -> ", similarity_score)
        # reiterate = False
        if similarity_score <= previous_similarity_score or lambda_val <= 6:
            print("Convereged with LAMBDA = ", lambda_val)
            reiterate = False
        else:
            lambda_val = lambda_val - 1
            previous_similarity_score = similarity_score
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<< SET NEW VALUE FOR LAMBDA >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("New Lambda = ", lambda_val)
