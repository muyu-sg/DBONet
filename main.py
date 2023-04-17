from __future__ import print_function, division
import random
from tqdm import tqdm
from util.clusteringPerformance import StatisticClustering
import numpy as np
import torch
import torch.nn as nn
import scipy.io
from model import DBONet
from util.utils import features_to_adj, normalization, standardization
import os


def train(features, adj, epoch, block, gamma, labels, n_view, n_clusters, model, optimizer, scheduler, device):
    acc_max = 0.0
    res = []

    # data tensor
    for i in range(n_view):
        exec("features_{}= torch.from_numpy(features[{}]/1.0).float().to(device)".format(i,i))
        exec("features_{}= standardization(normalization(features_{}))".format(i,i))
        exec("features[{}]= torch.Tensor(features[{}] / 1.0).to(device)".format(i,i))
        exec("adj[{}]=adj[{}].to_dense().float().to(device)".format(i, i))

    criterion = nn.MSELoss()
    with tqdm(total=epoch, desc="Training") as pbar:
        for i in range(epoch):
            model.train()
            optimizer.zero_grad()
            output_z = model(features, adj)


            loss_dis = torch.Tensor(np.array([0])).to(device)
            loss_lap = torch.Tensor(np.array([0])).to(device)

            for k in range(n_view):

                    exec("loss_dis+=criterion(output_z.mm(output_z.t()), features_{}.mm(features_{}.t()))".format(k, k))
                    exec("loss_lap+=criterion(output_z.mm(output_z.t()), adj[{}])".format(k, k))

            loss = loss_dis + gamma * loss_lap
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            Dis_loss = loss_dis.cpu().detach().numpy()
            Lap_loss = loss_lap.cpu().detach().numpy()
            train_loss = loss.cpu().detach().numpy()
            output_zz = output_z.detach().cpu().numpy()

            [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(output_zz, labels, n_clusters)
            if (ACC[0] > acc_max):
                acc_max = ACC[0]
                res = []
                for item in [ACC, NMI, Purity, ARI, Fscore, Precision, Recall]:
                    res.append("{}({})".format(item[0] * 100, item[1] * 100))
            pbar.update(1)
            print({"Dis_loss": "{:.6f}".format(Dis_loss[0]), "Lap_loss": "{:.6f}".format(Lap_loss[0]),
                   'Loss': '{:.6f}'.format(train_loss[0]),
                   'ACC': '{:.2f} | {:.2f}'.format(ACC[0] * 100, acc_max * 100)})
    return res


def getInitF(dataset, n_view, dataset_dir):
    dataset=dataset + "WG"
    data = scipy.io.loadmat(os.path.join(dataset_dir, dataset))
    Z = data[dataset]
    Z_init = Z[0][0]
    for i in range(1, Z.shape[1]):
        Z_init += Z[0][i]
    return Z_init / n_view


def main(data, args):
    # Clustering evaluation metrics
    SCORE = ['ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall']

    seed = args.seed
    block = args.block
    epoch = args.epoch
    thre = args.thre
    lr = args.lr
    gamma = args.gamma
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    adj, features, labels, nfeats, n_view, n_clusters = features_to_adj(data, args.path + args.data_path)


    n = len(adj[0])
    print("samples:{}, view size:{}, feature dimensions:{}, class:{}".format(n, n_view, nfeats, n_clusters))

    # initial representation
    Z_init = getInitF(data , n_view, args.path + args.data_path)

    # network architecture
    model = DBONet(nfeats, n_view, n_clusters, block, thre,  Z_init, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.90, 0.92), eps=0.01, weight_decay=0.15)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=15, verbose=True,
                                                           min_lr=1e-8)

    print("gamma:{}, block:{}, epoch:{}, thre:{}, lr:{}\n".format(
        gamma, block, epoch,  thre, lr))

    # Training
    res = train(features, adj, epoch, block, gamma, labels, n_view, n_clusters, model, optimizer, scheduler, device)

    print("{}:{}\n".format(data, dict(zip(SCORE, res))))
