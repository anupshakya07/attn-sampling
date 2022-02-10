import logging
import pandas as pd
import gensim.models
import time
import pickle

from sklearn.model_selection import train_test_split

BOS_TOKEN = 3
EOS_TOKEN = 4

class RelationWordEmbedding:
    def __init__(self, ip_path, embedding_dimension=300, sliding_window_size=8):
        self.input_file_path = ip_path
        self.embedding_dimension = embedding_dimension
        self.sliding_window_size = sliding_window_size
        self.sentences = []
        self.model = None

    def generate_sentences(self, tokenize_data=True, kc_maps=None):
        columns = ['Anon Student Id', 'Problem Hierarchy', 'Problem Name', 'KC(SubSkills)', 'Correct First Attempt']
        chunk_iterator = pd.read_csv(self.input_file_path, sep="\t", header=0, iterator=True, chunksize=1000000,
                                     usecols=columns)
        print("------Generating Sentences to train the word embeddings-------")
        start_time = time.time()
        output_data = []
        max_length = 0

        for chunk_data in chunk_iterator:
            filtered_chunk_data = chunk_data[chunk_data["KC(SubSkills)"].notnull()]
            for student_id, std_groups in filtered_chunk_data.groupby('Anon Student Id'):
                prob_hierarchy = std_groups.groupby('Problem Hierarchy')
                for hierarchy, hierarchy_groups in prob_hierarchy:
                    prob_name = hierarchy_groups.groupby('Problem Name')
                    for problem_name, prob_name_groups in prob_name:
                        sub_skills = prob_name_groups['KC(SubSkills)']
                        cfas = prob_name_groups['Correct First Attempt']
                        full_strategy = []
                        kcs_prob = [BOS_TOKEN]
                        cfa_list = [BOS_TOKEN]
                        for kc_row, cfa_row in zip(sub_skills, cfas):
                            if str(kc_row) != "nan":
                                correct_first_attempt = cfa_row
                                temp = kc_row.split("~~")
                                full_strategy.extend(temp)  # {KC1,KC2,....,KCn}
                                self.sentences.append(temp)
                                for kc in kc_row.split("~~"):
                                    kcs_prob.append(kc_maps["kc_index_map"][kc])
                                    cfa_list.append(correct_first_attempt)
                                    list_with_single_kc = [student_id, hierarchy, problem_name, kc]
                                    self.sentences.append(list_with_single_kc)  # {StdId, ProbHierarchy, ProblemName, KC}
                                    st_kc = [student_id, kc]
                                    self.sentences.append(st_kc)
                                    prob_kc = [problem_name, kc]
                                    self.sentences.append(prob_kc)
                        self.sentences.append(full_strategy)
                        if 1 < len(kcs_prob) <= 150:
                            cfa_list.append(EOS_TOKEN)
                            kcs_prob.append(EOS_TOKEN)
                            if len(kcs_prob) > max_length:
                                max_length = len(kcs_prob)
                            output_data.append({"src": kcs_prob, "target": cfa_list})
        end_time = time.time()
        print("Time Taken for generating the sentences = ", end_time - start_time)
        if tokenize_data:
            train, val = train_test_split(output_data, test_size=0.2, shuffle=True, random_state=40)
            data_dump = {
                "max_length": max_length,
                "train": train,
                "val": val,
                "kc_index_map": kc_maps["kc_index_map"],
                "index_kc_map": kc_maps["index_kc_map"],
                "BOS": 3,
                "EOS": 4
            }
            pickle.dump(data_dump, open("output/tokenized_data_BOS_EOS.pkl", "wb"))

    def train(self, tokenize_data, kc_maps):
        self.generate_sentences(tokenize_data=tokenize_data, kc_maps=kc_maps)
        print("******************************* Training Relational Word Embeddings *******************************")
        print("Parameters:->>>>>>>")
        print("Number of sentences trained = ", len(self.sentences))
        print("Embedding Dimension of the Word Embeddings = ", self.embedding_dimension)
        print("Sliding Window Size = ", self.sliding_window_size)
        start_time = time.time()
        self.model = gensim.models.Word2Vec(self.sentences, size=self.embedding_dimension, window=self.sliding_window_size, min_count=1)
        end_time = time.time()
        print("Time taken to train the word embeddings = ", end_time - start_time, " secs.")

    def save_trained_model(self, model_name):
        self.model.save(model_name)

    def load_trained_model(self, model_name):
        self.model = gensim.models.Word2Vec.load(model_name)