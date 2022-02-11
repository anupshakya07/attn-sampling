from src import data_preprocessor
from src.lstm_ns_clustered import NsClusteredModel
from src.one_hot_encoder import OneHotEnc
import pickle
import argparse
import gensim


def main(options):
    data_processor = pickle.load(open("output/processed_data.pkl", "rb"))
    embedding_model = gensim.models.Word2Vec.load(options.embedding_model)

    one_hot_encoder = OneHotEnc()
    one_hot_encoder.train(data_processor.unique_kcs)

    model = NsClusteredModel(200, len(data_processor.unique_kcs), options.dataset, embedding_model, one_hot_encoder)
    model.generate_training_sample(data_processor.unique_students, data_processor.unique_problems, 10, 100)
    model.train_model(num_epochs=10, batch_size=10)

    model.setup_inference_model()
    model.evaluate_training_accuracy(100)

    test_x, test_y, max_target_length = model.generate_sample(options.test_file_path, 200)
    model.evaluate_model(test_x, test_y, 3, max_target_length, one_hot_encoder.model, "Test")
    model.save_model("output/models/strategy_model/test-model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str,
                        default="dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_train.txt")

    parser.add_argument('-test_file_path', type=str,
                        default="dataset/bridge_to_algebra_2008_2009/sample_bridge_to_algebra_2008_2009_test.txt")

    parser.add_argument('-embedding_model', type=str,
                        default="embeddings-models/embedding-models/pruned-embedding-model-75-percent-retained")

    opt = parser.parse_args()

    main(opt)
