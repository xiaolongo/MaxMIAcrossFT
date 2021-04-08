import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch_geometric.utils import negative_sampling

from utils import uniform


class VGAE(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        self.mu, self.logstd = self.encoder(x, edge_index)
        self.logstd = self.logstd.clamp(max=10)
        z = self.reparametrize(self.mu, self.logstd)
        return z

    @staticmethod
    def decoder(z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def loss(self, z, edge_index):
        # recont loss
        pos_loss = -torch.log(self.decoder(z, edge_index) + 1e-7).mean()
        neg_edge_index = negative_sampling(edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index) +
                              1e-7).mean()
        recont_loss = pos_loss + neg_loss

        # kl loss
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + 2 * self.logstd - self.mu**2 - self.logstd.exp()**2,
                      dim=1))
        return recont_loss + (1 / z.size(0)) * kl_loss

    @staticmethod
    def test(train_z, train_y, test_z, test_y):
        clf = LogisticRegression(solver='lbfgs',
                                 multi_class='auto',
                                 max_iter=1000)
        clf.fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())


class DeepGraphInfomax(torch.nn.Module):
    def __init__(self, hidden_dim, encoder, summary, corruption):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.hidden_dim, self.weight)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        self.pos_z = self.encoder(x, edge_index)
        self.neg_z = self.encoder(self.corruption(x), edge_index)
        self.s_z = self.summary(self.pos_z)
        if not self.training:
            return self.pos_z

    def discriminator(self, z, s_z):
        value = torch.matmul(z, torch.matmul(self.weight, s_z))
        return torch.sigmoid(value)

    def loss(self):
        pos_loss_z = torch.log(
            self.discriminator(self.pos_z, self.s_z) + 1e-7).mean()
        neg_loss_z = torch.log(1 - self.discriminator(self.neg_z, self.s_z) +
                               1e-7).mean()
        mi_loss_z = pos_loss_z + neg_loss_z

        return -(mi_loss_z)

    @staticmethod
    def test(train_z, train_y, test_z, test_y):
        clf = LogisticRegression(solver='lbfgs',
                                 multi_class='auto',
                                 max_iter=1000)
        clf.fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())


class MultiViewDGI(torch.nn.Module):
    def __init__(self, hidden_dim, encoder, encoder_d, summary, corruption):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.encoder_d = encoder_d
        self.summary = summary
        self.corruption = corruption

        self.weight = torch.nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.weight_d = torch.nn.Parameter(torch.Tensor(
            hidden_dim, hidden_dim))

        self.mlp_node = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                      nn.PReLU(hidden_dim),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.PReLU(hidden_dim))

        self.mlp_graph = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.PReLU(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.PReLU(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.hidden_dim, self.weight)
        uniform(self.hidden_dim, self.weight_d)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # the first view
        self.pos_z = self.encoder(x, edge_index)
        self.neg_z = self.encoder(self.corruption(x), edge_index)
        self.s = self.summary(self.pos_z)

        # the second view
        self.pos_z_d = self.encoder_d(x, edge_index)
        self.neg_z_d = self.encoder_d(self.corruption(x), edge_index)
        self.s_d = self.summary(self.pos_z_d)

        if not self.training:
            return self.pos_z + self.pos_z_d

    def discriminator(self, z, s):
        value = torch.matmul(z, torch.matmul(self.weight, s))
        return torch.sigmoid(value)

    def discriminator_d(self, z, s):
        value = torch.matmul(z, torch.matmul(self.weight_d, s))
        return torch.sigmoid(value)

    def loss(self):
        # the first view
        pos_loss = torch.log(self.discriminator(self.pos_z, self.s_d) +
                             1e-7).mean()
        neg_loss = torch.log(1 - self.discriminator(self.neg_z, self.s_d) +
                             1e-7).mean()
        mi_loss = pos_loss + neg_loss

        # the second view
        pos_loss_d = torch.log(
            self.discriminator(self.pos_z_d, self.s) + 1e-7).mean()
        neg_loss_d = torch.log(1 - self.discriminator(self.neg_z_d, self.s) +
                               1e-7).mean()
        mi_loss_d = pos_loss_d + neg_loss_d

        return -(mi_loss + mi_loss_d)

    @staticmethod
    def test(train_z, train_y, test_z, test_y):
        clf = LogisticRegression(solver='lbfgs',
                                 multi_class='auto',
                                 max_iter=1000)
        clf.fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())


class MVMIFT(torch.nn.Module):
    def __init__(self, hidden_dim, encoder_f, encoder_t, encoder_c, summary,
                 corruption):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder_f = encoder_f
        self.encoder_t = encoder_t
        self.encoder_c = encoder_c
        self.summary = summary
        self.corruption = corruption

        self.weight_z_t = torch.nn.Parameter(
            torch.Tensor(hidden_dim, hidden_dim))
        self.weight_z_f = torch.nn.Parameter(
            torch.Tensor(hidden_dim, hidden_dim))
        self.weight_z_cf = torch.nn.Parameter(
            torch.Tensor(hidden_dim, hidden_dim))
        self.weight_z_ct = torch.nn.Parameter(
            torch.Tensor(hidden_dim, hidden_dim))

        self.mlp_ft = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.PReLU(hidden_dim),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.PReLU(hidden_dim))
        self.mlp_c = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                   nn.PReLU(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.PReLU(hidden_dim))
        self.mlp_s = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                   nn.PReLU(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.PReLU(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.hidden_dim, self.weight_z_t)
        uniform(self.hidden_dim, self.weight_z_f)
        uniform(self.hidden_dim, self.weight_z_cf)
        uniform(self.hidden_dim, self.weight_z_ct)

    def forward(self, data, feature_edge_index):
        x, edge_index = data.x, data.edge_index
        # feature view
        self.pos_z_f = self.encoder_f(x, feature_edge_index)
        self.neg_z_f = self.encoder_f(self.corruption(x), feature_edge_index)
        self.s_f = self.summary(self.pos_z_f)
        self.s_f = self.mlp_s(self.s_f.unsqueeze(0)).squeeze()
        self.pos_z_f = self.mlp_ft(self.pos_z_f)
        self.neg_z_f = self.mlp_ft(self.neg_z_f)

        # topology view
        self.pos_z_t = self.encoder_t(x, edge_index)
        self.neg_z_t = self.encoder_t(self.corruption(x), edge_index)
        self.s_t = self.summary(self.pos_z_t)
        self.s_t = self.mlp_s(self.s_t.unsqueeze(0)).squeeze()
        self.pos_z_t = self.mlp_ft(self.pos_z_t)
        self.neg_z_t = self.mlp_ft(self.neg_z_t)

        # common view
        self.pos_z_cf = self.encoder_c(x, feature_edge_index)
        self.pos_z_ct = self.encoder_c(x, edge_index)
        self.pos_z_cft = torch.cat([self.pos_z_cf, self.pos_z_ct], dim=-1)
        self.pos_z_cft = self.mlp_c(self.pos_z_cft)

        neg_z_cf = self.encoder_c(self.corruption(x), feature_edge_index)
        neg_z_ct = self.encoder_c(self.corruption(x), edge_index)
        self.neg_z_cft = torch.cat([neg_z_cf, neg_z_ct], dim=-1)
        self.neg_z_cft = self.mlp_c(self.neg_z_cft)

        self.s_cft = self.summary(self.pos_z_cft)

        if not self.training:
            inference_out = [self.pos_z_f, self.pos_z_t, self.pos_z_cft]
            return torch.stack(inference_out, dim=-1).mean(dim=-1)

    def discriminator_t(self, z, s):
        value = torch.matmul(z, torch.matmul(self.weight_z_t, s))
        return torch.sigmoid(value)

    def discriminator_f(self, z, s):
        value = torch.matmul(z, torch.matmul(self.weight_z_f, s))
        return torch.sigmoid(value)

    def discriminator_cf(self, z, s):
        value = torch.matmul(z, torch.matmul(self.weight_z_cf, s))
        return torch.sigmoid(value)

    @staticmethod
    def decoder(z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value)

    def recont_loss(self, z, edge_index):
        pos_edge_index = edge_index
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        pos_recont_loss = -torch.log(self.decoder(z, pos_edge_index) +
                                     1e-7).mean()
        neg_recont_loss = -torch.log(1 - self.decoder(z, neg_edge_index) +
                                     1e-7).mean()
        recont_loss = pos_recont_loss + neg_recont_loss
        return recont_loss

    def loss(self, data, feature_edge_index):
        # feature view
        pos_loss_f = torch.log(
            self.discriminator_f(self.pos_z_f, self.s_t) + 1e-7).mean()
        neg_loss_f = torch.log(1 -
                               self.discriminator_f(self.neg_z_f, self.s_t) +
                               1e-7).mean()
        mi_loss_f = pos_loss_f + neg_loss_f

        # topology view
        pos_loss_t = torch.log(
            self.discriminator_t(self.pos_z_t, self.s_f) + 1e-7).mean()
        neg_loss_t = torch.log(1 -
                               self.discriminator_t(self.neg_z_t, self.s_f) +
                               1e-7).mean()
        mi_loss_t = pos_loss_t + neg_loss_t

        # common view
        pos_loss_cf = torch.log(
            self.discriminator_cf(self.pos_z_cft, self.s_cft) + 1e-7).mean()
        neg_loss_cf = torch.log(
            1 - self.discriminator_cf(self.neg_z_cft, self.s_cft) +
            1e-7).mean()
        mi_loss_cf = pos_loss_cf + neg_loss_cf

        # recont loss
        recont_loss_cftf = self.recont_loss(self.pos_z_cft, feature_edge_index)
        recont_loss_cftt = self.recont_loss(self.pos_z_cft, data.edge_index)
        recont_loss = recont_loss_cftf + recont_loss_cftt

        # disagreement regularization
        cosine_loss_f = F.cosine_similarity(self.pos_z_f, self.pos_z_cf).mean()
        cosine_loss_t = F.cosine_similarity(self.pos_z_t, self.pos_z_ct).mean()
        cosine_loss = cosine_loss_f + cosine_loss_t

        return -(mi_loss_f + mi_loss_t + 0.5 *
                 (mi_loss_cf - recont_loss) - 1 * cosine_loss)

    @staticmethod
    def test(train_z, train_y, test_z, test_y):
        clf = LogisticRegression(solver='lbfgs',
                                 multi_class='auto',
                                 max_iter=10000)
        clf.fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())
