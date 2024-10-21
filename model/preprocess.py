from __future__ import absolute_import, division, print_function

import os
import gzip
import argparse

import pandas as pd

from data_utils import Dataset
from knowledge_graph import KnowledgeGraph
from likr_utils import DATASET_DIR, save_labels, ML1M, TMP_DIR, save_dataset, load_dataset, save_kg

def generate_labels(dataset, mode='train'):
    review_file = f"{DATASET_DIR[dataset]}/{mode}.txt.gz"
    user_products = {}  # {uid: [pid,...], ...}
    with gzip.open(review_file, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)
    save_labels(dataset, user_products, mode=mode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ml1m", help='ML1M')
    args = parser.parse_args()

    print('Load', args.dataset, 'dataset from file...')
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
    dataset = Dataset(args)
    save_dataset(args.dataset, dataset)

    print('Create', args.dataset, 'knowledge graph from dataset...')
    dataset = load_dataset(args.dataset)
    kg = KnowledgeGraph(dataset)
    kg.compute_degrees()
    save_kg(args.dataset, kg)

    print('Generate', args.dataset, 'train/test labels.')
    generate_labels(args.dataset, 'train')
    generate_labels(args.dataset, 'valid')
    generate_labels(args.dataset, 'test')

if __name__ == '__main__':
    main()
