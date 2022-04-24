import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate
from sklearn.linear_model import LogisticRegression
from .bdc_module import BDC


class STLDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(STLDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

        reduce_dim = params.reduce_dim
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
        self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)

        self.C = params.penalty_C
        self.params = params

    def feature_forward(self, x):
        out = self.dcov(x)
        return out

    def set_forward(self, x, is_feature=True):
        with torch.no_grad():
            z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.detach()
        z_query = z_query.detach()

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        qry_norm = torch.norm(z_query, p=2, dim=1).unsqueeze(1).expand_as(z_query)
        spt_norm = torch.norm(z_support, p=2, dim=1).unsqueeze(1).expand_as(z_support)
        qry_normalized = z_query.div(qry_norm + 1e-6)
        spt_normalized = z_support.div(spt_norm + 1e-6)

        z_query = qry_normalized.detach().cpu().numpy()
        z_support = spt_normalized.detach().cpu().numpy()
        y_support = np.repeat(range(self.n_way), self.n_support)

        clf = LogisticRegression(penalty='l2',
                                    random_state=0,
                                    C=self.C,
                                    solver='lbfgs',
                                    max_iter=1000,
                                    multi_class='multinomial')
        clf.fit(z_support, y_support)
        scores = clf.predict(z_query)

        return scores
