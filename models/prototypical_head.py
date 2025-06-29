# models/prototypical_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalHead(nn.Module):
    def __init__(self, metric="euclidean"):
        super().__init__()
        self.metric = metric

    def forward(self, query_embeddings, prototypes):
        if self.metric == "euclidean":
            # Compute -||x - c||^2
            n_query = query_embeddings.size(0)
            n_proto = prototypes.size(0)

            query = query_embeddings.unsqueeze(1)  # [Q, 1, D]
            proto = prototypes.unsqueeze(0)         # [1, P, D]
            logits = -torch.sum((query - proto) ** 2, dim=2)  # [Q, P]

        elif self.metric == "cosine":
            query = F.normalize(query_embeddings, dim=1)
            proto = F.normalize(prototypes, dim=1)
            logits = torch.mm(query, proto.t())  # [Q, P]

        else:
            raise ValueError("Unsupported metric")

        return logits
