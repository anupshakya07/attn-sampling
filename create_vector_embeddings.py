import argparse
import pickle
from src.data_preprocessor import DataPreprocessor
from src.obj2vec_embedder import RelationWordEmbedding
from src.one_hot_encoder import OneHotEnc

EMBEDDING_DIMENSION = 300


def main(opt):
    data_processor = DataPreprocessor(input_file_path=opt.dataset_path)
    data_processor.analyze_dataset()

    vector_embedder = RelationWordEmbedding(opt.dataset_path, embedding_dimension=EMBEDDING_DIMENSION,
                                            sliding_window_size=8)
    vector_embedder.train()
    vector_embedder.save_trained_model(opt.embedding_model_output)


    one_hot_encoder = OneHotEnc()
    one_hot_encoder.train(data_processor.unique_kcs)

    collect_vector_embeddings(vector_embedder.model, data_processor)


def collect_vector_embeddings(embedding_model, data_processor):
    unique_student_embeddings = []
    unique_problems_embeddings = []

    for student in data_processor.unique_students:
        if student in embedding_model.wv.vocab:
            unique_student_embeddings.append(embedding_model.wv[student])

    for problem in data_processor.unique_problems:
        if problem in embedding_model.wv.vocab:
            unique_problems_embeddings.append(embedding_model.wv[problem])

    pickle.dump(unique_student_embeddings, open("input/unique_student_embeddings.pkl", "wb"))
    pickle.dump(unique_problems_embeddings, open("input/unique_problems_embeddings.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_path', type=str,
                        default="dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt")
    parser.add_argument('-embedding_model_output', type=str,
                        default="embedding-models/gensim-bridge-to-algebra-embedding-model")

    options = parser.parse_args()
    main(options)
