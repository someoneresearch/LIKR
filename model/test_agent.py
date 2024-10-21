from __future__ import absolute_import, division, print_function

import os
import argparse
from math import log
import numpy as np
import torch
import json
from easydict import EasyDict as edict
from tqdm import tqdm
from functools import reduce
from kg_env import BatchKGEnvironment
from train_agent import ActorCritic
from likr_utils import *
import warnings
import csv

warnings.filterwarnings("ignore", category=DeprecationWarning)   

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evaluate(dataset_name, topk_matches, test_user_products, ks=[20, 40]):
    invalid_users = []
    metrics_by_k = {k: edict(ndcg=[], hr=[], precision=[], recall=[]) for k in ks}
    ndcgs = []
    test_user_idxs = list(test_user_products.keys())
    rel_size = []
    for uid in test_user_idxs:
        if uid not in topk_matches:
            invalid_users.append(uid)
            continue
        rel_set = test_user_products[uid]
        if len(rel_set) == 0:
            continue
        pred_list = topk_matches[uid][::-1]
        rel_size.append(len(rel_set))
        for k in ks:
            if len(pred_list) < k:
                continue
            top_k_preds = pred_list[:k]
            hit_num = 0.0
            hit_list = []
            for pid in top_k_preds:
                if pid in rel_set:
                    hit_num += 1
                    hit_list.append(1)
                else:
                    hit_list.append(0)
            ndcg = ndcg_at_k(hit_list, k)
            recall = hit_num / len(rel_set)
            precision = hit_num / len(top_k_preds)
            hit = 1.0 if hit_num > 0.0 else 0.0
            metrics_by_k[k].ndcg.append(ndcg)
            metrics_by_k[k].hr.append(hit)
            metrics_by_k[k].recall.append(recall)
            metrics_by_k[k].precision.append(precision)
    for k in ks:
        avg_metrics = edict(
            ndcg=[],
            hr=[],
            precision=[],
            recall=[],
        )
        print(f"Results for k={k}:")
        for metric, values in metrics_by_k[k].items():
            avg_metrics[metric] = np.mean(values)
            avg_metric_value = np.mean(values) * 100 if metric == "ndcg_other" else np.mean(values)
            n_users = len(values)
            print("Overall for noOfUser={}, {}={:.4f}".format(n_users, metric, avg_metric_value))
        print("\n")
    makedirs(dataset_name)
    with open(RECOM_METRICS_FILE_PATH[dataset_name], 'w') as f:
        json.dump(metrics_by_k, f)

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def batch_beam_search(env, model, uids, device, intrain=None, topk=[4, 2, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)
    state_pool = env.reset(uids)
    path_pool = env._batch_path
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)
        actmask_pool = _batch_acts_to_masks(acts_pool)
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))
        probs_max, _ = torch.max(probs, 0)
        probs_min, _ = torch.min(probs, 0)
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()
        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):
                    continue
                relation, next_node_id = acts_pool[row][idx]
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[env.dataset_name][path[-1][1]][
                        relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)
    return path_pool, probs_pool

def predict_paths(policy_file, path_file, args):
    print('Predicting paths...')
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len,
                             state_history=args.state_history)
    pretrain_sd = torch.load(policy_file)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())
    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs = batch_beam_search(env, model, batch_uids, args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))

def save_output(dataset_name, pred_paths):
    extracted_path_dir = LOG_DATASET_DIR[dataset_name]
    if not os.path.isdir(extracted_path_dir):
        os.makedirs(extracted_path_dir)
    print("Normalizing items scores...")
    score_list = []
    for uid, pid in pred_paths.items():
        for pid, path_list in pred_paths[uid].items():
            for path in path_list:
                score_list.append(float(path[0]))
    min_score = min(score_list)
    max_score = max(score_list)
    print("Saving pred_paths...")
    for uid in pred_paths.keys():
        curr_pred_paths = pred_paths[uid]
        for pid in curr_pred_paths.keys():
            curr_pred_paths_for_pid = curr_pred_paths[pid]
            for i, curr_path in enumerate(curr_pred_paths_for_pid):
                path_score = pred_paths[uid][pid][i][0]
                path_prob = pred_paths[uid][pid][i][1]
                path = pred_paths[uid][pid][i][2]
                new_path_score = (float(path_score) - min_score) / (max_score - min_score)
                pred_paths[uid][pid][i] = (new_path_score, path_prob, path)
    with open(extracted_path_dir + "/pred_paths.pkl", 'wb') as pred_paths_file:
        pickle.dump(pred_paths, pred_paths_file)
    pred_paths_file.close()

def extract_paths(dataset_name, save_paths, path_file, train_labels, valid_labels, test_labels):
    embeds = load_embed(args.dataset)
    user_embeds = embeds[USER]
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    product = main_entity
    watched_embeds = embeds[main_relation][0]
    movie_embeds = embeds[main_entity]
    scores = np.dot(user_embeds + watched_embeds, movie_embeds.T)
    validation_pids = get_validation_pids(dataset_name)
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {uid: {} for uid in test_labels}
    for path, probs in zip(results['paths'], results['probs']):
        if path[-1][1] != product:
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        if uid in valid_labels and pid in valid_labels[uid]:
            continue
        if pid in train_labels[uid]:
            continue
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))
    return pred_paths, scores

def evaluate_paths(dataset_name, pred_paths, emb_scores, train_labels, test_labels):
    best_pred_paths = {}
    for uid in pred_paths:
        if uid in train_labels:
            train_pids = set(train_labels[uid])
        else:
            print("Invalid train_pids")
        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0])
    sort_by = 'prob'
    pred_labels = {}
    pred_paths_top40 = {}
    for uid in best_pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)
        top40_pids = [p[-1][2] for _, _, p in sorted_path[:40]]
        top40_paths = [p for _, _, p in sorted_path[:40]]
        if args.add_products and len(top40_pids) < 40:
            train_pids = set(train_labels[uid])
            cand_pids = np.argsort(emb_scores[uid])
            for cand_pid in cand_pids[::-1]:
                if cand_pid in train_pids or cand_pid in top40_pids:
                    continue
                top40_pids.append(cand_pid)
                if len(top40_pids) >= 40:
                    break
        pred_labels[uid] = top40_pids[::-1]
        pred_paths_top40[uid] = top40_paths[::-1]
    evaluate(dataset_name, pred_labels, test_labels)

def get_path_pattern_weigth(path_pattern_name, pred_uv_paths):
    n_same_path_pattern = 0
    total_paths = len(pred_uv_paths)
    for path in pred_uv_paths:
        if path_pattern_name == get_path_pattern(path):
            n_same_path_pattern += 1
    return log(2 + (n_same_path_pattern / total_paths))

def test(args):
    policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)
    print(path_file)
    train_labels = load_labels(args.dataset, 'train')
    valid_labels = load_labels(args.dataset, 'valid')
    test_labels = load_labels(args.dataset, 'test')
    if args.run_path:
        predict_paths(policy_file, path_file, args)
    if args.save_paths or args.run_eval:
        pred_paths, scores = extract_paths(args.dataset, args.save_paths, path_file, train_labels, valid_labels, test_labels)
    if args.run_eval:
        evaluate_paths(args.dataset, pred_paths, scores, train_labels, test_labels)

if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {ml1m, lfm1m}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=400, help='Max number of actions.')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--add_products', type=boolean, default=True, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=list, nargs='*', default=[4, 2, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    parser.add_argument('--save_paths', type=boolean, default=True, help='Save paths')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    args.log_dir = os.path.join(TMP_DIR[args.dataset], args.name)
    test(args)

