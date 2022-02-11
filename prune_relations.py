from gensim import models
import pandas as pd
import pickle
import torch
from transformer.Models import Transformer
from transformer.Translator import Translator
import numpy as np
import argparse
from tqdm import tqdm
import gensim


def get_trimmed_kcs(kc_list, kc_cfa_model, device, index_kc_map, clip_keys, clip_list):
    kc_list.insert(0, 3)
    kc_list.append(4)
    kc_tensor = torch.tensor([kc_list], dtype=torch.int64).to(device)
    prediction, dec_enc_attentions = kc_cfa_model.translate_sentence(kc_tensor)
    attn_numpy = dec_enc_attentions[0][0][1:, 1:-1].cpu().detach().numpy()
    trimmed_kcs_dict = dict()

    for clip, key in zip(clip_list, clip_keys):
        clip_length = (int(len(attn_numpy[0]) // clip))
        clipped_kc_indices = np.argpartition(attn_numpy.mean(axis=0), -clip_length)[-clip_length:]
        trimmed_kcs = [index_kc_map[kc_list[idx + 1]] for idx in clipped_kc_indices]
        trimmed_kcs_dict[key] = trimmed_kcs

    return trimmed_kcs_dict


def main(options):
    std_ph_prob = []
    clip_keys = ["95-percent-retained", "90-percent-retained", "85-percent-retained", "80-percent-retained",
                 "75-percent-retained", "70-percent-retained"]
    all_clipped_kcs = {clip_keys[0]: [],
                       clip_keys[1]: [],
                       clip_keys[2]: [],
                       clip_keys[3]: [],
                       clip_keys[4]: [],
                       clip_keys[5]: []}
    clip_lengths_list = [20 / 19, 10 / 9, 20 / 17, 5 / 4, 4 / 3, 10 / 7]
    all_data = pickle.load(open(opt.data_path, "rb"))
    index_kc_map = all_data["index_kc_map"]
    kc_index_map = all_data["kc_index_map"]

    original_embedding_model = gensim.models.Word2Vec.load(options.embedding_model)

    if options.clip_kcs:

        n_src_vocab = len(kc_index_map) + 5
        max_length = 50
        device = torch.device('cuda' if options.use_gpu else 'cpu')

        transformer = Transformer(n_src_vocab=n_src_vocab,
                                  n_trg_vocab=2 + 3,
                                  src_pad_idx=2,
                                  trg_pad_idx=2,
                                  trg_emb_prj_weight_sharing=False,
                                  emb_src_trg_weight_sharing=False,
                                  n_position=300)

        checkpoint = torch.load(options.mastery_model)
        transformer.load_state_dict(checkpoint["model"])

        translator = Translator(model=transformer,
                                beam_size=5,
                                max_seq_len=max_length,
                                src_pad_idx=2,
                                trg_pad_idx=2,
                                trg_bos_idx=3,
                                trg_eos_idx=4
                                ).to(device=device)
        transformer.eval()

        data = pd.read_csv(options.dataset, sep="\t", header=0, chunksize=1000000, iterator=True)

        chunk_number = 1
        for chunk in data:
            iteration_desc = "Chunk " + str(chunk_number)
            filtered_data = chunk[chunk["KC(SubSkills)"].notnull()]
            student_group = filtered_data.groupby('Anon Student Id')

            with tqdm(total=len(student_group.groups), desc=iteration_desc) as progress_bar:
                for student_id, std_groups in student_group:
                    prob_hierarchy = std_groups.groupby('Problem Hierarchy')
                    for hierarchy, hierarchy_groups in prob_hierarchy:
                        prob_name = hierarchy_groups.groupby('Problem Name')
                        for problem_name, prob_name_groups in prob_name:
                            subskills = prob_name_groups['KC(SubSkills)']
                            all_kcs = []
                            for a in subskills:
                                if str(a) != "nan":
                                    temp = a.split("~~")
                                    for kc in temp:
                                        if kc in kc_index_map.keys():
                                            all_kcs.append(kc_index_map[kc])
                            if all_kcs:
                                clipped_kcs_dict = get_trimmed_kcs(all_kcs, translator, device, index_kc_map, clip_keys,
                                                                   clip_lengths_list)
                                std_ph_prob.append([student_id, hierarchy, problem_name])
                                for key, val in clipped_kcs_dict.items():
                                    all_clipped_kcs[key].append(val)
                    progress_bar.update(1)
                progress_bar.close()
                chunk_number += 1

        pickle.dump(std_ph_prob, open("output/std_ph_prob_ablation.pkl", "wb"))
        pickle.dump(all_clipped_kcs, open("output/all_clipped_kcs_ablation.pkl", "wb"))

    else:
        std_ph_prob = pickle.load(open("output/std_ph_prob_ablation.pkl", "rb"))
        all_clipped_kcs = pickle.load(open("output/all_clipped_kcs_ablation.pkl", "rb"))

    if options.train_embeddings:

        for clip_key in clip_keys:
            sentences = []
            for inp, kc_seq in zip(std_ph_prob, all_clipped_kcs[clip_key]):
                for kc in kc_seq:
                    sentences.append([inp[0], inp[1], inp[2], kc])
                    sentences.append([inp[0], kc])
                    sentences.append([inp[1], kc])
                    sentences.append([inp[2], kc])
                sentences.append(kc_seq)
            print("Length of Sentences for ", clip_key, " = ", len(sentences))

            embedding_model = models.Word2Vec(sentences, size=300, window=4, min_count=1)
            model_path = "embedding-models/embedding-model-" + clip_key

            embedding_model.save(model_path)

            if options.prune_embeddings:
                std_cluster = pickle.load(open("output/student_cluster.pkl", "rb"))
                prob_cluster = pickle.load(open("output/problem_cluster.pkl", "rb"))
                cluster_attention_scores = pickle.load(open("output/cluster_attention_scores.pkl", "rb"))

                data_processor = pickle.load(open("output/processed_data.pkl", "rb"))
                unique_students = data_processor.unique_students
                unique_problems = data_processor.unique_problems

                std_list = [embedding_model.wv.most_similar(positive=[x], topn=1)[0][0] for x in std_cluster.X]
                prob_list = [embedding_model.wv.most_similar(positive=[x], topn=1)[0][0] for x in prob_cluster.X]

                print("Number of unique students = ", len(std_list))
                print("Number of unique problems = ", len(prob_list))

                for student in tqdm(unique_students, desc="Student Loop - "):
                    if student in std_list:
                        cluster_id = std_cluster.dataClusterId[std_list.index(student)]
                        all_cluster_keys = list(cluster_attention_scores.keys())
                        matched_cluster_keys = []
                        matched_cluster_attention_score = []

                        for key in all_cluster_keys:
                            if key.split(",")[0] == str(cluster_id):
                                matched_cluster_keys.append(key)
                        for matched_key in matched_cluster_keys:
                            matched_cluster_attention_score.append(cluster_attention_scores[matched_key])
                        avg_attention_score = np.array(matched_cluster_attention_score).mean()
                        if student in original_embedding_model.wv.vocab and student in embedding_model.wv.vocab:
                            new_embedding = np.add(original_embedding_model.wv[student] * avg_attention_score,
                                                   embedding_model.wv[student] * (1 - avg_attention_score))

                            # embedding_model.wv.add_vector(student, new_embedding)
                            original_embedding_model.wv.add([student], [new_embedding], replace=True)

                for problem in tqdm(unique_problems, desc=" Problem Loop - "):
                    if problem in prob_list:
                        cluster_id = prob_cluster.dataClusterId[prob_list.index(problem)]
                        all_cluster_keys = list(cluster_attention_scores.keys())
                        matched_cluster_keys = []
                        matched_cluster_attention_score = []

                        for key in all_cluster_keys:
                            if key.split(",")[1] == str(cluster_id):
                                matched_cluster_keys.append(key)
                        for matched_key in matched_cluster_keys:
                            matched_cluster_attention_score.append(cluster_attention_scores[matched_key])
                        avg_attention_score = np.array(matched_cluster_attention_score).mean()
                        if problem in original_embedding_model.wv.vocab and problem in embedding_model.wv.vocab:
                            new_embedding = np.add(original_embedding_model.wv[problem] * avg_attention_score,
                                                   embedding_model.wv[problem] * (1 - avg_attention_score))

                            # embedding_model.wv.add_vector(student, new_embedding)
                            original_embedding_model.wv.add([problem], [new_embedding], replace=True)

                pruned_model_name = "embedding-models/pruned-embedding-model-" + clip_key
                embedding_model.save(pruned_model_name)
                print("Output Embedding Model saved at : ", pruned_model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-use_gpu', type=bool, default=True)

    parser.add_argument('-clip_kcs', type=bool, default=True)

    parser.add_argument('-train_embeddings', type=bool, default=True)

    parser.add_argument('-prune_embeddings', type=bool, default=True)

    parser.add_argument('-mastery_model', type=str, default="output/models/mastery_model/model.chkpt")

    parser.add_argument('-dataset', type=str,
                        default="dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt")

    parser.add_argument('-data_path', type=str, default="output/tokenized_data_BOS_EOS.pkl")

    parser.add_argument('-embedding_model', type=str,
                        default="embeddings-models/gensim-bridge-to-algebra-embedding-model")

    opt = parser.parse_args()

    main(opt)
