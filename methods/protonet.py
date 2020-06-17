import torch.nn as nn
from utils import euclidean_dist, Model_type

class ProtoNet(nn.Module):

    def __init__(self, encoder, args):
        super().__init__()
        self.args = args
        self.encoder = encoder

    def forward(self, data, mode = 'test'):
        args = self.args
        data = data.view(args.n_way*(args.n_shot + args.n_query), *data.size()[2:])
        feature = self.encoder(data)

        feature = feature.view(args.n_way, args.n_shot+args.n_query, -1)
        z_support = feature[:, :args.n_shot]
        z_query = feature[:, args.n_shot:]

        proto = z_support.view(args.n_way, args.n_shot, -1).mean(1)
        z_query = z_query.contiguous().view(args.n_way*args.n_query, -1)
        logits = -euclidean_dist(z_query, proto) / self.args.temperature
        return logits