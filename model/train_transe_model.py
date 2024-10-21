from __future__ import absolute_import, division, print_function

import os
import argparse
import torch
import torch.optim as optim
from data_utils import DataLoader
from likr_utils import *
from transe_model import KnowledgeEmbedding

logger = None

def train(args, dataset):
    dataloader = DataLoader(dataset, args.batch_size)
    review_to_train = len(dataset.review.data) * args.epochs + 1

    model = KnowledgeEmbedding(args, dataloader).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    steps = 0
    smooth_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            lr = args.lr * max(1e-4, 1.0 - dataloader.finished_review_num / float(review_to_train))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            batch_idxs = dataloader.get_batch()
            batch_idxs = torch.from_numpy(batch_idxs).to(args.device)

            optimizer.zero_grad()
            train_loss = model(batch_idxs)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            smooth_loss += train_loss.item() / args.steps_per_checkpoint

            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                logger.info('Epoch: {:02d} | '.format(epoch) +
                            'Review: {:d}/{:d} | '.format(dataloader.finished_review_num, review_to_train) +
                            'Lr: {:.5f} | '.format(lr) +
                            'Smooth loss: {:.5f}'.format(smooth_loss))
                smooth_loss = 0.0
        if epoch % 10 == 0:
            torch.save(model.state_dict(), '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, epoch))


def extract_embeddings(args, dataset):

    dataset_name = args.dataset

    model_file = '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, args.epochs)
    print('Load embeddings', model_file)
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)

    embeds = {}

    for entity_name in dataset.entity_names:
        embeds[entity_name] = state_dict[f'{entity_name}.weight'].cpu().data.numpy()[:-1]

    embeds[INTERACTION[dataset_name]] = (
        state_dict[INTERACTION[dataset_name]].cpu().data.numpy()[0],
        state_dict[f'{INTERACTION[dataset_name]}_bias.weight'].cpu().data.numpy()
    )

    for relation_name in dataset.other_relation_names:
        embeds[relation_name] = (
            state_dict[f'{relation_name}'].cpu().data.numpy()[0],
            state_dict[f'{relation_name}_bias.weight'].cpu().data.numpy()
        )

    save_embed(dataset_name, embeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML1M, help='One of {ml1m, lfm1m}.')
    parser.add_argument('--name', type=str, default='train_transe_model', help='models name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
    parser.add_argument('--embed_size', type=int, default=100, help='knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
    args = parser.parse_args()

    os.makedirs(LOG_DATASET_DIR[args.dataset], exist_ok=True)
    with open(os.path.join(LOG_DATASET_DIR[args.dataset], f'{TRANSE_HPARAMS_FILE}'), 'w') as f:
        import json
        import copy
        args_dict = dict()
        for x,y in copy.deepcopy(args._get_kwargs()):
            args_dict[x] = y
        if 'device' in args_dict:
            del args_dict['device']
        json.dump(args_dict,f)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    print(TMP_DIR[args.dataset])
    args.log_dir = os.path.join(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    dataset = load_dataset(args.dataset)
    train(args, dataset)
    extract_embeddings(args, dataset)

if __name__ == '__main__':
    main()

