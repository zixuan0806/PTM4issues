
import gensim.downloader as api

from gensim.models.keyedvectors import KeyedVectors



if __name__ == "__main__":
    name = "word2vec-google-news-300"
    path = api.load(name, return_path=True)
    print(path)
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    model.save_word2vec_format(f'embedding/word2vec/{name}.txt', binary=False)