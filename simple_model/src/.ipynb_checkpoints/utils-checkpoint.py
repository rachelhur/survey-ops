import pickle
import os


def save_results(results: dict, fp: str):
    if not os.path.exists:
        os.makedirs('results')
    with open(fp, 'wb') as handle:
        pickle.dump(results, fp)

def load_results(fp: str):
    with open(fp, 'rb') as handle:
        return pickle.load(fp)