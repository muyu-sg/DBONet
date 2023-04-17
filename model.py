import torch
import torch.nn.functional as F
import torch.nn as nn


class Block(nn.Module):
    def __init__(self,  out_features, nfea,device):
        super(Block, self).__init__()
        self.S_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.S = nn.Linear(out_features, out_features).to(device)

        self.U_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.U = nn.Linear(nfea, out_features).to(device)

        self.device = device
    def forward(self, input, adj, view):
        input1 = self.S(self.S_norm(input))
        input2 = self.U(self.U_norm(view))
        output = torch.mm(adj, input)
        output = input1 + input2 - output
        return output

class DBONet(nn.Module):
    def __init__(self, nfeats, n_view,n_clusters, blocks, para, Z_init, device):
        super(DBONet, self).__init__()
        self.n_clusters = n_clusters
        self.blocks = blocks
        self.device=device
        self.n_view=n_view
        for i in range(n_view):
            exec('self.block{}=Block(n_clusters,{},device)'.format(i, nfeats[i] ))
        self.Z_init = torch.from_numpy(Z_init).float().to(device)
        self.theta = nn.Parameter(torch.FloatTensor([para]), requires_grad=True).to(device)

    def soft_threshold(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)

    def forward(self, features, adj):
        output_z=self.Z_init
        for i in range(0, self.blocks):
            z=torch.zeros_like(self.Z_init).to(self.device)
            for j in range(0,self.n_view):
                exec('z+=self.block{}(output_z, adj[{}], features[{}] )'.format(j,j,j))
            output_z=self.soft_threshold(z / self.n_view)
        return output_z
