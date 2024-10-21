from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from likr_utils import *
from data_utils import Dataset

class KnowledgeEmbedding(nn.Module):
    def __init__(self, args, dataloader):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda
        self.dataset_name = args.dataset
        self.relation_names = dataloader.dataset.other_relation_names
        self.entity_names = dataloader.dataset.entity_names
        self.relation2entity = dataloader.dataset.relation2entity
        self.initialize_entity_embeddings(dataloader.dataset)
        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)
        self.initialize_relations_embeddings(dataloader.dataset)
        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def initialize_entity_embeddings(self, dataset):
        self.entities = edict()
        for entity_name in self.entity_names:
            value = edict(vocab_size=getattr(dataset, entity_name).vocab_size)
            self.entities[entity_name] = value

    def initialize_relations_embeddings(self, dataset):
        self.relations = edict()
        main_rel = INTERACTION[dataset.dataset_name]
        self.relations[main_rel] = edict(
            et=PRODUCT,
            et_distrib=self._make_distrib(getattr(dataset, "review").product_uniform_distrib)
        )
        for relation_name in dataset.other_relation_names:
            value = edict(
                et=dataset.relation2entity[relation_name],
                et_distrib=self._make_distrib(getattr(dataset, relation_name).et_distrib)
            )
            self.relations[relation_name] = value

    def _entity_embedding(self, vocab_size):
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size):
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        distrib = np.power(np.array(distrib, dtype=float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss(self, batch_idxs):
        regularizations = []
        user_idxs = batch_idxs[:, 0]
        product_idxs = batch_idxs[:, 1]
        knowledge_relations = get_knowledge_derived_relations(self.dataset_name)
        up_loss, up_embeds = self.neg_loss(USER, INTERACTION[self.dataset_name], PRODUCT, user_idxs, product_idxs)
        regularizations.extend(up_embeds)
        loss = up_loss
        i = 2
        for curr_rel in knowledge_relations:
            entity_name, curr_idxs = self.relation2entity[curr_rel], batch_idxs[:, i]
            curr_loss, curr_embeds = self.neg_loss(PRODUCT, curr_rel, entity_name, product_idxs, curr_idxs)
            if curr_loss is not None:
                regularizations.extend(curr_embeds)
                loss += curr_loss
            i += 1
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss
        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):
        mask = entity_tail_idxs >= 0
        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []
        entity_head_embedding = getattr(self, entity_head)
        entity_tail_embedding = getattr(self, entity_tail)
        relation_vec = getattr(self, relation)
        relation_bias_embedding = getattr(self, relation + '_bias')
        entity_tail_distrib = self.relations[relation].et_distrib
        return kg_neg_loss(entity_head_embedding, entity_tail_embedding,
                        fixed_entity_head_idxs, fixed_entity_tail_idxs,
                        relation_vec, relation_bias_embedding, self.num_neg_samples, entity_tail_distrib)


def kg_neg_loss(entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib):
    batch_size = entity_head_idxs.size(0)
    entity_head_vec = entity_head_embed(entity_head_idxs)
    example_vec = entity_head_vec + relation_vec
    example_vec = example_vec.unsqueeze(2)
    entity_tail_vec = entity_tail_embed(entity_tail_idxs)
    pos_vec = entity_tail_vec.unsqueeze(1)
    relation_bias = relation_bias_embed(entity_tail_idxs).squeeze(1)
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias
    pos_loss = -pos_logits.sigmoid().log()
    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)
    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]
