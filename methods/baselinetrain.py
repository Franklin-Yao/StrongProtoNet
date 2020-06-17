import torch.nn as nn
from utils import Model_type, euclidean_dist

# --- conventional supervised training ---
class BaselineTrain(nn.Module):
  def __init__(self, encoder, args, loss_type = 'softmax'):
    super().__init__()
    self.encoder = encoder
    self.args = args
    if args.model_type is Model_type.ResNet12:
      final_feat_dim = 640
    else:
      pass
    if args.dataset == 'MiniImageNet':
      n_class = 64
    else:
      pass
    self.classifier = nn.Linear(final_feat_dim, n_class)
    self.classifier.bias.data.fill_(0)
    self.loss_fn = nn.CrossEntropyLoss()

  def forward(self,data, mode='train'):
    args = self.args
    if mode in ['val', 'test']:
      data = data.view(args.n_way * (args.n_shot + args.n_query), *data.size()[2:])
      feature  = self.encoder(data)
      feature = feature.view(args.n_way, args.n_shot + args.n_query, -1)
      z_support = feature[:, :args.n_shot]
      z_query = feature[:, args.n_shot:]

      proto = z_support.view(args.n_way, args.n_shot, -1).mean(1)
      z_query = z_query.contiguous().view(args.n_way * args.n_query, -1)
      scores = -euclidean_dist(z_query, proto) / self.args.temperature
    else:
      feature = self.encoder(data)
      return feature
      # scores  = self.classifier(feature)
    return scores
