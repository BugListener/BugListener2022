import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GraphConv, global_max_pool, global_add_pool
from transformers import BertModel


def graph_construction(features, role, seq_length, graph_edge, attention, ifcuda):
    speaker_type_map = [[0, 1], [2, 3]]  # 四种关系0,0->0; 0,1->1; 1,0->2; 1,1->3
    assert len(graph_edge) == features.size(0)
    node_features, edge_index, edge_weight, edge_type, graph_batch = [], [], [], [], []

    scores = attention(features, graph_edge)
    batch_size = features.size(0)
    #  将一个batch中的所有图组合成一个大图
    node_sum = 0
    for i in range(batch_size):
        node_features.append(features[i, :seq_length[i], :])

        for src_node, tgt_node in graph_edge[i]:
            edge_index.append(torch.tensor([src_node + node_sum, tgt_node + node_sum]))
            edge_weight.append(scores[i, src_node, tgt_node])

            src_role = role[i][src_node]
            tgt_role = role[i][tgt_node]
            edge_type.append(speaker_type_map[src_role][tgt_role])

        node_sum += seq_length[i]
        graph_batch.append(torch.tensor([i] * seq_length[i]))

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).t().contiguous()
    edge_weight = torch.stack(edge_weight)
    edge_type = torch.tensor(edge_type)
    graph_batch = torch.cat(graph_batch)

    if ifcuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_weight = edge_weight.cuda()
        edge_type = edge_type.cuda()
        graph_batch = graph_batch.cuda()

    return node_features, edge_index, edge_weight, edge_type, graph_batch


class EdgeAttention(nn.Module):
    def __init__(self, feature_dim, ifcuda):
        super(EdgeAttention, self).__init__()
        self.transformation = nn.Linear(feature_dim, feature_dim, bias=False)
        self.ifcuda = ifcuda

    def forward(self, features, graph_edge):
        # features -> batch_size * seq_num * embedd_dim
        # scores -> batch * seq_num * seq_num
        scores = torch.matmul(features, self.transformation(features).transpose(-2, -1))
        # mask disconnected edge
        graph_edge_ = []
        for i, j in enumerate(graph_edge):
            for x in j:
                graph_edge_.append([i, x[0], x[1]])
        graph_edge_ = np.array(graph_edge_).transpose()
        if self.ifcuda:
            mask = torch.zeros(scores.size()).detach().cuda()
        else:
            mask = torch.zeros(scores.size()).detach()
        mask[graph_edge_] = 1
        scores = scores.masked_fill(mask == 0, float("-inf"))

        scores = F.softmax(scores, dim=1)
        return scores


class GraphNetwork(torch.nn.Module):
    def __init__(self, feature_dim, graph_class_num, num_relations, hidden_size=64, dropout=0.5, ifcuda=True):
        super(GraphNetwork, self).__init__()

        self.conv1 = GraphConv(feature_dim, hidden_size)
        self.conv2 = RGCNConv(hidden_size, hidden_size, num_relations, num_bases=30)
        self.linear = nn.Linear(2 * (feature_dim + hidden_size), hidden_size)
        self.graph_smax_fc = nn.Linear(hidden_size, graph_class_num)
        self.dropout = nn.Dropout(dropout)
        self.ifcuda = ifcuda

    def forward(self, x, edge_index, edge_weight, edge_type, graph_batch):
        # x -> node_num * feature_dim
        out = self.conv1(x, edge_index=edge_index, edge_weight=edge_weight)
        out = out.relu()
        out = self.conv2(x=out, edge_index=edge_index, edge_type=edge_type)
        out = out.relu()
        # out -> node_num * hidden_size
        # features -> node_num * (feature_dim + hidden_size)
        features = torch.cat([x, out], dim=-1)
        # max_feature -> batch * (feature_dim + hidden_size)
        max_features = global_max_pool(features, graph_batch)
        # sum_features -> batch * (feature_dim + hidden_size)
        sum_features = global_add_pool(features, graph_batch)
        hidden = F.relu(self.linear(self.dropout(torch.cat([sum_features, max_features], dim=-1))))
        hidden = self.graph_smax_fc(self.dropout(hidden))
        graph_log_prob = F.log_softmax(hidden, -1)

        return graph_log_prob


class BugListener(nn.Module):

    def __init__(self, pretrained_model, D_bert, filter_sizes, filter_num, D_cnn, D_graph, n_speakers,
                 graph_class_num=2, dropout=0.5, ifcuda=True):
        super(BugListener, self).__init__()
        self.ifcuda = ifcuda
        self.pretrained_bert = BertModel.from_pretrained(pretrained_model)
        self.cnn_encoder = nn.ModuleList([nn.Conv1d(D_bert, filter_num, size) for size in filter_sizes])
        self.fc_cnn = nn.Linear(len(filter_sizes) * filter_num, D_cnn)
        self.att_model = EdgeAttention(D_cnn, self.ifcuda)
        n_relations = n_speakers ** 2
        self.graph_net = GraphNetwork(D_cnn, graph_class_num, n_relations, D_graph, dropout, self.ifcuda)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask, role, seq_length, graph_edge):
        # 1. utterance embedding
        b_, s_, w_ = input_ids.size()
        # -> (batch*sen_num) * word_num
        i_ids = input_ids.view(-1, w_)
        t_ids = token_type_ids.view(-1, w_)
        a_ids = attention_mask.view(-1, w_)

        # word_output = (batch*sen_num) * word_num * D_bert
        word_output = self.pretrained_bert(input_ids=i_ids, token_type_ids=t_ids, attention_mask=a_ids)[0]
        # 对padding的向量进行mask
        mask = a_ids.unsqueeze(-1).expand_as(word_output)  # mask -> (batch*sen_num) * word_num * D_bert
        word_output = word_output.masked_fill(mask == 0, 0.)
        # -> (batch*sen_num) * dim * word_num
        word_output = word_output.transpose(-2, -1).contiguous()

        convoluted = [F.relu(conv(word_output)) for conv in self.cnn_encoder]
        pooled = [F.max_pool1d(c, c.size(-1)).squeeze(-1) for c in convoluted]
        concated = torch.cat(pooled, -1)
        features = F.relu(self.fc_cnn(self.dropout(concated)))  # (num_utt * batch, dim) -> (num_utt * batch, dim)
        features = features.view(b_, s_, -1)  # (num_utt * batch, D_cnn) -> (batch, num_utt, D_cnn)
        # 2. graph construction
        node_features, edge_index, edge_weight, edge_type, graph_batch = \
            graph_construction(features, role, seq_length, graph_edge, self.att_model, self.ifcuda)

        # 3. graph embedding and classification
        graph_log_prob = self.graph_net(node_features, edge_index, edge_weight, edge_type, graph_batch)
        return graph_log_prob
