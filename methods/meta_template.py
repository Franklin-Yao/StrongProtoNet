import torch.nn as nn
import torch
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter

class MetaTemplate(nn.Module):
  def __init__(self, model_func, exp_tag, n_way, n_support, flatten=True, leakyrelu=False, tf_path=None, change_way=True):
    super(MetaTemplate, self).__init__()
    self.exp_tag = exp_tag
    self.n_way      = n_way
    self.n_support  = n_support
    self.n_query    = -1 #(change depends on input)
    self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
    self.feat_dim   = self.feature.final_feat_dim
    self.change_way = change_way  #some methods allow different_way classification during training and test
    self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

  @abstractmethod
  def set_forward(self,x,is_feature):
    pass

  @abstractmethod
  def forward(self, x):
    pass

  # def forward(self,x):
  #   out  = self.feature.forward(x)
  #   return out

  def parse_feature(self,x,is_feature):
    x = x.cuda()
    if is_feature:
      z_all = x
    else:
      x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
      z_all       = self.feature.forward(x)
      z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
    z_support   = z_all[:, :self.n_support]
    z_query     = z_all[:, self.n_support:]

    return z_support, z_query

  def correct(self, x):
    scores, loss = self.forward(x)
    # y_query = np.repeat(range( self.n_way ), self.n_query )
    y_query = np.repeat(range(self.n_way), self.n_query)

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == y_query)
    return float(top1_correct), len(y_query), loss.item()*len(y_query)

  def train_loop(self, epoch, train_loader, optimizer, total_it):
    print_freq = len(train_loader) // 10
    avg_loss=0
    for i, (x,_ ) in enumerate(train_loader):

      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      if self.exp_tag in ['2domain']:
        self.n_way = int(self.n_way/2)
        support_set = x[self.n_way:, :self.n_support]
        query_set = x[:self.n_way, :self.n_query]
        x = torch.cat((support_set, query_set), dim=1)
      optimizer.zero_grad()
      _, loss = self.forward(x)
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss+loss.item()

      if (i + 1) % print_freq==0:
        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar(self.method + '/query_loss', loss.item(), total_it + 1)
      total_it += 1
    return total_it

  def trainall_loop(self, epoch, ps_loader, pu_loader, optimizer, total_it):
    print_freq = len(ps_loader) / 10
    avg_model_loss = 0.
    avg_ft_loss = 0.

    print_freq = len(ps_loader) // 10
    avg_loss = 0
    for i, ((support, _), (query, _)) in enumerate(zip(ps_loader, pu_loader)):
      x = torch.cat((support, query), 1)
      self.n_query = x.size(1) - self.n_support

      if self.change_way:
        self.n_way = x.size(0)
      optimizer.zero_grad()
      _, loss = self.forward(x)
      loss.backward()
      optimizer.step()
      avg_loss = avg_loss + loss.item()

      if (i + 1) % print_freq == 0:
        print(
          'Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(ps_loader), avg_loss / float(i + 1)))
      if (total_it + 1) % 10 == 0 and self.tf_writer is not None:
        self.tf_writer.add_scalar(self.method + '/query_loss', loss.item(), total_it + 1)
      total_it += 1
    return total_it

  def test_loop(self, test_loader, record = None):
    loss = 0.
    count = 0
    acc_all = []

    iter_num = len(test_loader)
    for i, (x,_) in enumerate(test_loader):
      self.n_query = x.size(1) - self.n_support
      if self.change_way:
        self.n_way  = x.size(0)
      correct_this, count_this, loss_this = self.correct(x)
      acc_all.append(correct_this/ count_this*100 )
      loss += loss_this
      count += count_this

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('---Loss = %4.2f ---Test Acc = %4.2f +- %4.2f' %(loss/count, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

    return acc_mean
