import numpy as np


def load_official_test_data(data_dir="data/"):
    tiles = np.load(f"{data_dir}/tiles.npy")
    solution_lengths = np.load(f"{data_dir}/solution_lengths.npy")
    return tiles, solution_lengths
