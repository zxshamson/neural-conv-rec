import os
import sys
import random
import time
import torch
import argparse
import numpy as np
from torch import nn, optim
import torch.utils.data as data
import torch.nn.functional as F
from corpus import Corpus
from data_process import MyDataset, form_dataset, create_embedding_matrix, create_conv_data, create_arc_info
from rank_eval import cal_map, cal_ndcg_all, cal_precision_N
from Models import NCF, NCFBiLSTM, NCFGModel


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("modelname", type=str, choices=["NCF", "NCFBiLSTM", "NCFGCN", "NCFGRN"])
    parser.add_argument("--cuda_dev", type=str, default="0")
    parser.add_argument("--max_word_num", type=int, default=-1)
    parser.add_argument("--pred_pc", type=float, default=0.75)
    parser.add_argument("--factor_dim", type=int, default=20)
    parser.add_argument("--text_factor_dim", type=int, default=100)
    parser.add_argument("--neg_sample_num", type=int, default=5)
    parser.add_argument("--kernal_num", type=int, default=150)
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epoch", type=int, default=199)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mlp_layers_num", type=int, default=3)
    parser.add_argument("--gcn_layers_num", type=int, default=1)
    parser.add_argument("--grn_states_num", type=int, default=6)
    parser.add_argument("--runtime", type=int, default=0)
    parser.add_argument("--pos_weight", type=float, default=100)
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--bi_direction", action="store_true")
    parser.add_argument("--use_gates", action="store_true")
    parser.add_argument("--use_lstm", action="store_true")
    parser.add_argument("--ncf_pretrained", action="store_true")

    return parser.parse_args()


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * target * torch.log(torch.clamp(output, min=1e-15, max=1)) + \
            weights[0] * (1 - target) * torch.log(torch.clamp(1 - output, min=1e-15, max=1))
    else:
        loss = target * torch.log(torch.clamp(output, min=1e-15, max=1)) + \
            (1 - target) * torch.log(torch.clamp(1 - output, min=1e-15, max=1))

    return torch.neg(torch.mean(loss))


def evaluate(model, data, dev=False):
    model.eval()
    labels_list = {}
    for step, one_data in enumerate(data):
        label = one_data[-1].data.numpy()
        predictions = model(one_data[0])
        if torch.cuda.is_available():  # run in GPU
            pred_label = predictions.cpu().data.numpy()
            uc_pairs = one_data[0].cpu().data.numpy()
        else:
            pred_label = predictions.data.numpy()
            uc_pairs = one_data[0].data.numpy()
        for n in xrange(len(label)):
            try:
                labels_list[uc_pairs[n, 0]].append((pred_label[n], label[n]))
            except KeyError:
                labels_list[uc_pairs[n, 0]] = [(pred_label[n], label[n])]
    res_map = cal_map(labels_list)
    if dev:
        return res_map
    else:
        res_p1 = cal_precision_N(labels_list, 1)
        res_p5 = cal_precision_N(labels_list, 5)
        res_ndcg5 = cal_ndcg_all(labels_list, 5)
        res_ndcg10 = cal_ndcg_all(labels_list, 10)
        return res_map, res_p1, res_p5, res_ndcg5, res_ndcg10


def train_epoch(model, train_data, loss_weights, optimizer, epoch):
    model.train()
    start = time.time()
    print('Epoch: %d start!' % epoch)
    avg_loss = 0.0
    count = 0
    for step, one_data in enumerate(train_data):
        label = one_data[-1]
        if torch.cuda.is_available():  # run in GPU
            label = label.cuda()
        predictions = model(one_data[0])
        loss = weighted_binary_cross_entropy(predictions, label, loss_weights)
        avg_loss += loss.item()
        count += 1
        if count % 50000 == 0:
            print('Epoch: %d, iterations: %d, loss: %g' % (epoch, count, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        # if clip != -1:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # print 'step'
    avg_loss /= len(train_data)
    end = time.time()
    print('Epoch: %d done! Train avg_loss: %g! Using time: %.2f minutes!' % (epoch, avg_loss, (end-start)/60))
    return avg_loss


def train(config):
    filename = config.filename
    modelname = config.modelname
    corp = Corpus(filename, config.max_word_num, config.pred_pc)
    config.user_num, config.conv_num, config.vocab_num = corp.userNum, corp.convNum, corp.wordNum
    config.sampling = True if 'reddit' in filename else False
    train_ratings, test_data, dev_data = form_dataset(corp, config.batch_size, sampling=config.sampling)
    if modelname != "NCF":
        conv_data = create_conv_data(corp.convs)
        if config.ncf_pretrained:
            ncf_path = "BestModels/NCF/For_pretrained/" + filename.split(".")[0] + "/" + str(config.factor_dim) + \
                       "_" + str(config.text_factor_dim) + "_" + str(config.pred_pc) + ".model"
            if torch.cuda.is_available():  # run in GPU
                weights = torch.load(ncf_path)
            else:
                weights = torch.load(ncf_path, map_location='cpu')
            ncf_weights = (weights['mf_user_embedding.weight'], weights['mf_conv_embedding.weight'], weights['mlp_user_embedding.weight'])
        else:
            ncf_weights = None
        if modelname == "NCFGCN" or modelname == "NCFGRN":
            arcs = create_arc_info(conv_data)
    if modelname == "NCF":
        embedding_matrix = None
    else:
        embedding_matrix = create_embedding_matrix(filename, corp, config.embedding_dim)

    if modelname == 'NCF':
        model = NCF(config)
        path_name = str(config.pred_pc) + "_" + str(config.batch_size) + "_" + str(config.factor_dim) + "_" + \
            str(config.text_factor_dim) + "_" + str(config.mlp_layers_num) + "_" + str(config.neg_sample_num) + "_" + \
            str(config.lr) + "_" + str(int(config.pos_weight)) + "-" + str(config.runtime)
    elif modelname == 'NCFBiLSTM':
        model = NCFBiLSTM(config, conv_data, embedding_matrix, ncf_weights)
        path_name = str(config.pred_pc) + "_" + str(config.batch_size) + "_" + str(config.factor_dim) + "_" + \
            str(config.text_factor_dim) + "_" + str(config.kernal_num) + "_" + str(config.hidden_dim) + "_" + \
            str(config.mlp_layers_num) + "_" + str(config.neg_sample_num) + "_" + str(config.lr) + "_" + \
            str(int(config.pos_weight)) + "_" + ("bi" if config.bi_direction else "nbi") + "_" + \
            ("pt" if config.ncf_pretrained else "npt") + "-" + str(config.runtime)
    elif modelname == 'NCFGCN':
        model = NCFGModel(config, modelname, conv_data, arcs, embedding_matrix, ncf_weights)
        path_name = str(config.pred_pc) + "_" + str(config.batch_size) + "_" + str(config.factor_dim) + "_" + \
            str(config.text_factor_dim) + "_" + str(config.kernal_num) + "_" + str(config.hidden_dim) + "_" + \
            str(config.mlp_layers_num) + "_" + str(config.gcn_layers_num) + "_" + str(config.neg_sample_num) + \
            "_" + str(config.lr) + "_" + str(int(config.pos_weight)) + "_" + ("g" if config.use_gates else "ng") + "_" + \
            ("lstm" if config.use_lstm else "nlstm") + "_" + ("pt" if config.ncf_pretrained else "npt") + "-" + str(config.runtime)
    elif modelname == 'NCFGRN':
        model = NCFGModel(config, modelname, conv_data, arcs, embedding_matrix, ncf_weights)
        path_name = str(config.pred_pc) + "_" + str(config.batch_size) + "_" + str(config.factor_dim) + "_" + \
            str(config.text_factor_dim) + "_" + str(config.kernal_num) + "_" + str(config.hidden_dim) + "_" + \
            str(config.mlp_layers_num) + "_" + str(config.grn_states_num) + "_" + str(config.neg_sample_num) + "_" + \
            str(config.lr) + "_" + str(int(config.pos_weight)) + "_" + ("pt" if config.ncf_pretrained else "npt") + "-" + str(config.runtime)
    else:
        print 'Modelname Wrong!'
        exit()

    res_path = "BestResults/" + modelname + "/" + filename.split('.')[0] + "/"
    mod_path = "BestModels/" + modelname + "/" + filename.split('.')[0] + "/"
    if not os.path.isdir(res_path):
        os.makedirs(res_path)
    if not os.path.isdir(mod_path):
        os.makedirs(mod_path)
    mod_path += path_name + '.model'
    res_path += path_name + '.data'

    loss_weights = torch.Tensor([1, config.pos_weight])
    if torch.cuda.is_available():  # run in GPU
        model = model.cuda()
        loss_weights = loss_weights.cuda()
    if config.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / ((epoch + 1) ** 0.5))
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch // 10))

    best_dev_map = -1.0
    best_epoch = -1
    no_improve = 0
    for epoch in range(config.max_epoch):
        scheduler.step()
        train_data = MyDataset(corp, train_ratings, config.neg_sample_num, True)
        train_loader = data.DataLoader(train_data, batch_size=config.batch_size, num_workers=0, shuffle=True)
        loss = train_epoch(model, train_loader, loss_weights, optimizer, epoch)
        dev_map = evaluate(model, dev_data, dev=True)
        if dev_map > best_dev_map:
            no_improve = 0
            best_dev_map = dev_map
            os.system('rm ' + mod_path)
            best_epoch = epoch
            print('New Best Dev!!! MAP: %g' % best_dev_map)
            torch.save(model.state_dict(), mod_path)
        else:
            no_improve += 1
            print('Current Best Dev MAP: %g, Dev MAP: %g' % (best_dev_map, dev_map))
        if no_improve > 8:
            break
    model.load_state_dict(torch.load(mod_path))
    res = evaluate(model, test_data)
    print('Result in test set: MAP: %g, Precision@1: %g, Precision@5: %g, nDCG@5: %g, nDCG@10: %g' %
          (res[0], res[1], res[2], res[3], res[4]))
    with open(res_path, 'w') as f:
        f.write('MAP: %g, Precision@1: %g, Precision@5: %g, nDCG@5: %g, nDCG@10: %g\n' %
                (res[0], res[1], res[2], res[3], res[4]))
        f.write('Dev MAP: %g\n' % best_dev_map)
        f.write('Best epoch: %d\n' % best_epoch)


if __name__ == '__main__':
    config = parse_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_dev
    train(config)


