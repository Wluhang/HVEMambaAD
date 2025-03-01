import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBlock(nn.Module):
    def __init__(self, d_model, nodevec, conv_channel, skip_channel,
                 gcn_depth, dropout, propalpha, node_dim):
        super(GraphBlock, self).__init__()
        self.nodes = d_model
        self.nodevec = nodevec
        self.nodeveck = nn.Parameter(torch.randn(1, node_dim), requires_grad=True)
        self.start_conv = nn.Conv2d(1, conv_channel, (1, 1))
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, 1, (1, 1))
        self.norm = nn.LayerNorm(d_model)
        self.k = 4

    def forward(self, x):
        nodevec1 = self.nodevec * self.nodeveck # [8, 32]
        adj0 = F.softmax(F.relu(torch.mm(nodevec1, nodevec1.T)), dim=1)
        mask = torch.zeros(self.nodes, self.nodes).to(x.device)
        mask.fill_(float('0'))
        s1, t1 = adj0.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj0 * mask
        out = x.unsqueeze(1).transpose(2, 3)
        out = self.start_conv(out)
        out = self.gelu(self.gconv1(out, adj))
        out = self.end_conv(out).transpose(2, 3).squeeze()
        return self.norm(x + out)


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho