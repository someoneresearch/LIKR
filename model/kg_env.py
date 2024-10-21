from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import random
import os

from likr_utils import load_kg, load_embed, USER, SELF_LOOP, load_labels, MAIN_PRODUCT_INTERACTION, PATH_PATTERN, \
    KG_RELATION

class KGState(object):
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len
        if history_len == 0:
            self.dim = 2 * embed_size
        elif history_len == 1:
            self.dim = 4 * embed_size
        elif history_len == 2:
            self.dim = 6 * embed_size
        else:
            raise Exception('history length should be one of {0, 1, 2}')

    def __call__(self, user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                 older_relation_embed):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
        elif self.history_len == 2:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                                   older_relation_embed])
        else:
            raise Exception('mode should be one of {full, current}')


class BatchKGEnvironment(object):
    def __init__(self, dataset_str, max_acts, max_path_len=3, state_history=1, optimize_for=None, alpha=None, user_metadata_folder='newid_output'):
        self.max_acts = max_acts
        self.act_dim = max_acts + 1
        self.max_num_nodes = max_path_len + 1
        self.kg = load_kg(dataset_str)
        self.embeds = load_embed(dataset_str)
        self.embed_size = self.embeds[USER].shape[1]
        self.embeds[SELF_LOOP] = (np.zeros(self.embed_size), 0.0)
        self.state_gen = KGState(self.embed_size, history_len=state_history)
        self.state_dim = self.state_gen.dim
        self.dataset_name = dataset_str
        self.train_labels = load_labels(dataset_str, 'train')

        main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_str]
        u_p_scores = np.dot(self.embeds[USER] + self.embeds[main_relation][0], self.embeds[main_entity].T)
        self.u_p_scales = np.max(u_p_scores, axis=1)

        self.patterns = []
        valid_patterns = list(PATH_PATTERN[self.dataset_name].keys())

        for pattern_id in valid_patterns:
            pattern = PATH_PATTERN[self.dataset_name][pattern_id]
            pattern = [SELF_LOOP] + [v[0] for v in pattern[1:]]
            if len(pattern) == 3:
                pattern.append(SELF_LOOP)
            self.patterns.append(tuple(pattern))

        self._batch_path = None
        self._batch_curr_actions = None
        self._batch_curr_state = None
        self._batch_curr_reward = None
        self._done = False
        self.user_metadata = self._load_user_metadata(user_metadata_folder)

    def _load_user_metadata(self, folder_path):
        user_metadata = {}
        for filename in os.listdir(folder_path):
            if filename.startswith('user_') and filename.endswith('.txt'):
                user_id = int(filename[5:-4])
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    preferences = set()
                    for line in lines:
                        line = line.strip()
                        if line:
                            meta_type, meta_id = line.split(': ')
                            meta_id = int(meta_id)
                            preferences.add((meta_type, meta_id))
                user_metadata[user_id] = preferences
        return user_metadata

    def _has_pattern(self, path):
        pattern = tuple([v[0] for v in path])
        return pattern in self.patterns

    def _batch_has_pattern(self, batch_path):
        return [self._has_pattern(path) for path in batch_path]

    def _get_actions(self, path, done):
        main_product, review_interaction = MAIN_PRODUCT_INTERACTION[self.dataset_name]

        _, curr_node_type, curr_node_id = path[-1]
        actions = [(SELF_LOOP, curr_node_id)]

        if done:
            return actions

        relations_nodes = self.kg(curr_node_type, curr_node_id)
        candidate_acts = []
        visited_nodes = set([(v[1], v[2]) for v in path])
        for r in relations_nodes:
            next_node_type = KG_RELATION[self.dataset_name][curr_node_type][r]
            next_node_ids = relations_nodes[r]
            next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]
            candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))

        if len(candidate_acts) == 0:
            return actions

        if len(candidate_acts) <= self.max_acts:
            candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            return actions

        uid = path[0][-1]
        user_embed = self.embeds[USER][uid]

        scores = []
        for r, next_node_id in candidate_acts:
            next_node_type = KG_RELATION[self.dataset_name][curr_node_type][r]
            if next_node_type == USER:
                src_embed = user_embed
            elif next_node_type == main_product:
                src_embed = user_embed + self.embeds[review_interaction][0]
            else:
                src_embed = user_embed + self.embeds[main_product][0] + self.embeds[r][0]
            
            score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
            
            if curr_node_type == main_product and next_node_type in ['actor', 'writer', 'cinematographer', 'composer', 'director', 'editor', 'prodcompany', 'country', 'producer']:
                score += 10
            
            scores.append(score)
        candidate_idxs = np.argsort(scores)[-self.max_acts:]
        candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1]))
        actions.extend(candidate_acts)
        return actions

    def _batch_get_actions(self, batch_path, done):
        return [self._get_actions(path, done) for path in batch_path]

    def _get_state(self, path):
        user_embed = self.embeds[USER][path[0][-1]]
        zero_embed = np.zeros(self.embed_size)
        if len(path) == 1:
            state = self.state_gen(user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed)
            return state

        older_relation, last_node_type, last_node_id = path[-2]
        last_relation, curr_node_type, curr_node_id = path[-1]
        curr_node_embed = self.embeds[curr_node_type][curr_node_id]
        last_node_embed = self.embeds[last_node_type][last_node_id]
        last_relation_embed, _ = self.embeds[last_relation]
        if len(path) == 2:
            state = self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, zero_embed,
                                   zero_embed)
            return state

        _, older_node_type, older_node_id = path[-3]
        older_node_embed = self.embeds[older_node_type][older_node_id]
        older_relation_embed, _ = self.embeds[older_relation]
        state = self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed,
                               older_relation_embed)
        return state

    def _batch_get_state(self, batch_path):
        batch_state = [self._get_state(path) for path in batch_path]
        return np.vstack(batch_state)

    def _get_reward(self, path):
        uid = path[0][-1]
        if len(path) <= 2 or not self._has_pattern(path):
            return 0.0

        target_score = 0.0
        LLM_score = 0.0
        LLM_alpha = 0.6
        _, curr_node_type, curr_node_id = path[-1]
        main_entity, main_relation = MAIN_PRODUCT_INTERACTION[self.dataset_name]
        _, metadata_node_type, metadata_node_id = path[-2]

        user_preferences = self.user_metadata.get(uid, set())

        if (metadata_node_type, metadata_node_id) in user_preferences:
            if metadata_node_type == 'actor':
                LLM_score = 1
            elif metadata_node_type == 'writer':
                LLM_score = 1
            elif metadata_node_type == 'cinematographer':
                LLM_score = 1
            elif metadata_node_type == 'composer':
                LLM_score = 1
            elif metadata_node_type == 'country':
                LLM_score = 1
            elif metadata_node_type == 'director':
                LLM_score = 1
            elif metadata_node_type == 'editor':
                LLM_score = 1
            elif metadata_node_type == 'prodcompany':
                LLM_score = 1
            elif metadata_node_type == 'producer':
                LLM_score = 1
        
        if curr_node_type == main_entity:
            relation, shared_entity, eid = path[2]
            u_vec = self.embeds[USER][uid] + self.embeds[main_relation][0]
            p_vec = self.embeds[main_entity][curr_node_id]
            score = (np.dot(u_vec, p_vec) / self.u_p_scales[uid])
            target_score = max(score, 0.0)

        reward_score = target_score + LLM_alpha * LLM_score
        if reward_score > 99:
            print(reward_score)

        return reward_score

    def _batch_get_reward(self, batch_path):
        batch_reward = [self._get_reward(path) for path in batch_path]
        return np.array(batch_reward)

    def _is_done(self):
        return self._done or len(self._batch_path[0]) >= self.max_num_nodes

    def reset(self, uids=None):
        if uids is None:
            all_uids = list(self.kg(USER).keys())
            uids = [random.choice(all_uids)]

        self._batch_path = [[(SELF_LOOP, USER, uid)] for uid in uids]
        self._done = False
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state

    def batch_step(self, batch_act_idx):
        assert len(batch_act_idx) == len(self._batch_path)

        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            _, curr_node_type, curr_node_id = self._batch_path[i][-1]
            relation, next_node_id = self._batch_curr_actions[i][act_idx]
            if relation == SELF_LOOP:
                next_node_type = curr_node_type
            else:
                next_node_type = KG_RELATION[self.dataset_name][curr_node_type][relation]
            self._batch_path[i].append((relation, next_node_type, next_node_id))

        self._done = self._is_done()
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path)

        return self._batch_curr_state, self._batch_curr_reward, self._done

    def batch_action_mask(self, dropout=0.0):
        batch_mask = []
        for actions in self._batch_curr_actions:
            act_idxs = list(range(len(actions)))
            if dropout > 0 and len(act_idxs) >= 5:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = [act_idxs[0]] + tmp
            act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)

    def print_path(self):
        for path in self._batch_path:
            msg = 'Path: {}({})'.format(path[0][1], path[0][2])
            for node in path[1:]:
                msg += ' =={}=> {}({})'.format(node[0], node[1], node[2])
            print(msg)

