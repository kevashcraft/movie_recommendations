#!/usr/bin/env python
"""
Movie recommendations using lightfm

Siraj video source: https://www.youtube.com/watch?v=9gBC9R-msAk&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU&index=3

Created by: Kevin Ashcraft <kevin@kevashcraft.com>
Created on: 2018-03-05
"""
import argparse
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

def main():
    args = get_arguments()
    data = fetch_movielens(min_rating=4.0)
    model = train_model(data['train'])
    sample_recommendations(model, data['train'], data['item_labels'], args.users.split(','))

def get_arguments():
    parser = argparse.ArgumentParser(description='Movie Recommendations')
    parser.add_argument('--users', help='UserIDs to create recommendations for', required=True)
    return parser.parse_args()

def train_model(data):
    model = LightFM(loss='warp')
    model.fit(data, epochs=30, num_threads=2)

    return model

def sample_recommendations(model, data, labels, user_ids):
    n_users, n_items = data.shape

    for user_id in user_ids:
        known_positives = labels[data.tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = labels[np.argsort(-scores)]

        print("User {}".format(user_id))
        print("     Known positives:")

        for x in known_positives[:3]:
            print("         {}".format(x))

        print("     Recommended:")

        for x in top_items[:3]:
            print("         {}".format(x))

if __name__ == '__main__':
    main()

