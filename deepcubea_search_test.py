"""Test script for DeepCubeA Weighted A* Search.

Usage:
    uv run python deepcubea_search_test.py

Prerequisites:
    - Set DEEPCUBEA_MODEL_PATH in config.py to the trained model .pt file
    - Or pass it as command line argument: uv run python deepcubea_search_test.py <model_path>
"""
import sys
import time
from config import DEEPCUBEA_MODEL_PATH
from deepcubea_search import load_model, evaluate

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEEPCUBEA_MODEL_PATH
    if not model_path:
        print("ERROR: DEEPCUBEA_MODEL_PATH is empty.")
        print("Set it in config.py or pass as argument:")
        print("  uv run python deepcubea_search_test.py <model_path>")
        sys.exit(1)

    t0 = time.perf_counter()
    model = load_model(model_path)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s\n")

    evaluate(model, model_path=model_path)
