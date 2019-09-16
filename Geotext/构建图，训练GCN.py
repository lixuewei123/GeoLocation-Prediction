# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:47:48 2019

@author: GIS
"""

import pandas as pd
import numpy as np


user_df = pd.read_csv('E:\\deep_network\\Geotext dataset\\code\\GCN\\9236user_state_content_aite.csv',
                      encoding='ISO-8859-1')
aite = list(user_df.aite)
content = list(user_df.content)
state = list(user_df.state_one)


# ==================== 构建图 ==================================================
import networkx as nx
import pickle


def efficient_collaboration_weighted_projected_graph2(B, nodes):  # 输入图g， 0:60000
    nodes = set(nodes)          # nodes去重复，0，1,2,3...60000
    G = nx.Graph()
    G.add_nodes_from(nodes)		# 添加60000多个节点ID 节点为0-60000
    all_nodes = set(B.nodes())  # B图中所有的节点，60000多个用户 + @用户的非名人节点 0-6w+
    i = 0
    tenpercent = len(all_nodes) / 10  # 每10次，日志记录一次
    for m in all_nodes:			# g图中所有的节点，包括60000多个用户和@用户
        if i % tenpercent == 0:
            print(str(10 * i / tenpercent) + "%")
        i += 1  # 记录到第i个节点了

        nbrs = B[m]  # 可以看在g图中，m节点所有的邻居
        target_nbrs = [t for t in nbrs if t in nodes]  # 如果m的邻居都在nodes（60000用户）中，就添加到列表target_nbrs
        if m in nodes:  			# 如果m节点也在nodes中
            for n in target_nbrs:  	# m的在nodes中的邻居id， 也就是在nodes中的节点以及邻居，复制他们之间的边
                if m < n:			# 如果m<n， 建立边，没有这个判定条件也无所谓，减少了一半的运算量。
                    if not G.has_edge(m, n):	# 如果m和n之间没有边
                        G.add_edge(m, n)		# 建立无向无权边
		# m的所有在nodes中的邻居之间，也建立边。
        for n1 in target_nbrs:				# 在nodes中，m的邻居用户
            for n2 in target_nbrs:			# 在nodes中，m的其他邻居用户
                if n1 < n2:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
    return G



def construct_graph(user_df, celebrity_threshold=5):
    g = nx.Graph()
    
    user_df.aite = user_df.aite.fillna('')
    aite = list(user_df.aite)
    aite = [ele.replace('@','').replace('＠','').split(',') for ele in aite]
    mentions = aite
    
    nodes = list(user_df.uid)
    node_id = {node: ID for ID, node in enumerate(nodes)}
    with open('node_id.pkl', 'wb') as f:
        pickle.dump(node_id, f)
    
    g.add_nodes_from(list(node_id.values()))  # 用户ID对应的id数字作为节点
    for node in nodes:
        g.add_edge(node_id[node], node_id[node])  # 添加自环
    print(g.number_of_nodes(), g.number_of_edges())
    # nodes,mentions是一一对应的
    
    # 添加所有的边
    for i in range(len(nodes)):
        user = nodes[i]                 # 第i个用户的用户ID
        user_id = node_id[user]         # user_id = 第i个用户的ID
        mention = mentions[i]           # 第i个用户@的对象
        mention = list(set(mention))    # 第i个用户@ 的对象用户ID
        idmentions = set()
        for m in mention:               # 第i个用户@的每一个对象
            if m in node_id:            # 如果这个@对象，也在用户集里面
                idmentions.add(node_id[m])  # idmentions添加这个@对象的ID
            else:
                ID = len(node_id)       # 如果这个@对象不在用户集里面。
                node_id[m] = ID         # 在node_id中添加这个用户
                idmentions.add(ID)      # idmentions添加这个@对象的ID
        if len(idmentions) > 0:
            g.add_nodes_from(idmentions)  # 把这些被@到的用户也添加到图的节点
        for ID in idmentions:              # 第i个用户@的所有对象id
            g.add_edge(ID, user_id)        # 添加@对象和第i个用户的边

    celebrities = []  # 名人集
    for i in range(len(nodes), len(node_id)):  # nodes_list9000多个用户，node_id9000+@用户
        deg = len(g[i])  # 被@用户的邻居信息
        if deg == 1 or deg > celebrity_threshold:
            celebrities.append(i)  # 如果这个被@的用户是孤立的或者邻居的数量大于10
    g.remove_nodes_from(celebrities)  # 移除名人节点，以及所有的边
    print(g.number_of_nodes(), g.number_of_edges())
    
    projected_g = efficient_collaboration_weighted_projected_graph2(g, list(range(len(nodes))))  # 最终图。
    print(projected_g.number_of_nodes(), projected_g.number_of_edges())
    
    return projected_g

# 生成指定名人节点的图。
G = construct_graph(user_df, celebrity_threshold=5)
G.nodes()


# ========================== 生成邻接矩阵 ======================================
import scipy as sp

def product_adj(G):
    adj = nx.adjacency_matrix(G, nodelist=range(len(list(G.nodes()))), weight='w')  # 获取邻接矩阵
    adj.setdiag(0)              # 对角线元素设置为0
    selfloop_value = 1
    adj.setdiag(selfloop_value)  # 对角线元素设置为1
    n, m = adj.shape  # adj的维度
    diags = adj.sum(axis=1).flatten()  # 返回每列的和，向量
    with sp.errstate(divide='ignore'):
        diags_sqrt = 1.0 / sp.sqrt(diags)
    diags_sqrt[sp.isinf(diags_sqrt)] = 0
    D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    A = D_pow_neghalf * adj * D_pow_neghalf
    A = A.astype('float32')  # 得到拉普拉斯矩阵A
    return A

A = product_adj(G)
with open('E:\\deep_network\\Geotext dataset\\code\\GCN\\A.pkl', 'wb') as f:
    pickle.dump(A, f)


# ========================== 文本特征处理 ======================================
# ========================== doc2vec ======================================
import gensim
from gensim.models import Doc2Vec
import time
import random

def labelizeContent(content, label_type):  # 这是常用的加label函数了，你在网上搜也是差不多这样的。
    labelized = []
    for i, v in enumerate(content):
        label = '%s_%s' % (label_type, i)
        labelized.append(gensim.models.doc2vec.TaggedDocument(v, [label])) #TaggedDocument 与 LabeledSentence是一样效果的，后者是基于前者的。
    return labelized

def getVecs(model, corpus):
    # vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    # return np.concatenate(vecs)
    vecs = []
    for text in corpus:
        tmp = [model[w] for w in text.words]
        tmp = np.array(tmp)
        vecs.append(tmp.sum(axis=0))
    return np.array(vecs)


def train_doc2vec(user_df):
    text = user_df.content.values
    text = [sen.split() for sen in text]
    text = labelizeContent(text, 'train')
    data = text[:]
    model_dbow = Doc2Vec(dm=0, min_count=1, window=10, size=500, sample=1e-3, negative=5, workers=3)
    model_dbow.build_vocab(data[:])
    for epoch in range(20):
        all_reviews = data[:]
        random.shuffle(all_reviews)
        t_epoch = time.time()
        model_dbow.train(text, total_examples=model_dbow.corpus_count, epochs=3)  # epochs 设置为 1
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha
        print('=' * 30 + str(epoch) + '训练模型{:.4f}mins'.format((time.time() - t_epoch) / 60) + '=' * 30)
        model_dbow.save("E:\\deep_network\\Geotext dataset\\code\\GCN\\dbow.model")
    return model_dbow, data

model_dbow, data = train_doc2vec(user_df)
d2v = getVecs(model_dbow, data)
with open('E:\\deep_network\\Geotext dataset\\code\\GCN\\d2v.pkl', 'wb') as f:
    pickle.dump(d2v, f)


# ========================== tfidf ======================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus = list(user_df.stopword)
# step 1
vectoerizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')
# step 2
vectoerizer.fit(corpus)
# step 3
bag_of_words = vectoerizer.get_feature_names()
#print("Bag of words:")
#print(bag_of_words)
#print(len(bag_of_words))
# step 4
X = vectoerizer.transform(corpus)
print("index of `a` is : {}".format(vectoerizer.vocabulary_.get('aa')))

# step 1
tfidf_transformer = TfidfTransformer()
# step 2
tfidf_transformer.fit(X.toarray())
# step 3
#for idx, word in enumerate(vectoerizer.get_feature_names()):
#  print("{}\t{}".format(word, tfidf_transformer.idf_[idx]))
# step 4
tfidf = tfidf_transformer.transform(X)
print(tfidf.toarray().shape)
tfidf = tfidf.toarray()




# =========================== label ===========================================
labels = list(user_df.state_one)
label_map = {j: i for i, j in enumerate(list(set(labels)))}
labels = np.array(list(map(label_map.get, labels)))
#labels = labels + 1

data = [A, d2v, labels]
data = [A, tfidf, labels]
with open('E:\\deep_network\\Geotext dataset\\code\\GCN\\A_d2v_labels.pkl', 'wb') as f:
    pickle.dump(data, f)


# =========================== GCN Model ========================================
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features      # 输入特征数大小
        self.out_features = out_features    # 输出大小
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # dropout 在model中
        # Batch Nomorlization也是在model中
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, batch_normalization=False, highway=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.do_bn = batch_normalization        # 是否层标准化
        self.bn_input_gc1 = nn.BatchNorm1d(num_features=nhid, momentum=0.1)
        self.bn_input_gc2 = nn.BatchNorm1d(num_features=nhid, momentum=0.1)
        self.highway = highway

    def forward(self, x, adj):
        x = self.gc1(x, adj)                                    # gcn卷积
        if self.do_bn: x = self.bn_input_gc1(x)                 # BN
        x = F.relu(x)                                           # relu激活
        # if self.highway:
        #     x = highway_dense(x)                                # highway_dense层
        x = F.dropout(x, self.dropout, training=self.training)  # dropout
        x = self.gc2(x, adj)
        if self.do_bn: x = self.bn_input_gc2(x)                 # 第二层gcn的BN
        return F.log_softmax(x, dim=1)


def accuracy(output, labels):
    # output 2708*7 按照行选取最大值2708个 ,提取2708个最大值的位置（0-7）
    preds = output.max(1)[1].type_as(labels)  # 2708维，表示每个节点的分类位置
    correct = preds.eq(labels).double()     # preds 2708维，labels 2708维  对比pred-labels对应位置的值对不对
    correct = correct.sum()                 # 预测正确的数量
    return correct / len(labels)            # 返回预测正确率


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# === 准备输入数据 ===
A, d2v, labels = data
A = sparse_mx_to_torch_sparse_tensor(A)
X = sparse_mx_to_torch_sparse_tensor(tfidf)
X = torch.Tensor(d2v)
Y = torch.LongTensor(labels)

rows = A.shape[0]
idx_train = range(int(rows*0.6))
idx_val = range(int(rows*0.6), int(rows*0.8))
idx_test = range(int(rows*0.8), rows)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# 模型参数设置
lr = 0.001
input_size = X.shape[1]
output_size = len(set(list(labels)))
dropout = 0.5
regul = 0.000001
clf = GCN(nfeat=input_size, nhid=800, nclass=output_size, dropout=dropout, batch_normalization=False,
                      highway=False)
optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=regul)

# cuda
clf.cuda()
X = X.cuda()
A = A.cuda()
Y = Y.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()

print(X.shape, A.shape, Y.shape)


def train():
    EPOCH = 120
    print("迭代次数为 %d" % EPOCH)
    for epoch in range(EPOCH):
        t = time.time()
        clf.train()
        optimizer.zero_grad()
        output = clf(X, A)
        loss_train = F.nll_loss(output[idx_train], Y[idx_train])
        acc_train = accuracy(output[idx_train], Y[idx_train])
        loss_train.backward()
        optimizer.step()
        loss_val = F.nll_loss(output[idx_val], Y[idx_val])  # 验证集的损失
        acc_val = accuracy(output[idx_val], Y[idx_val])  # 验证集的准确度
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

def test():
    clf.eval()  # 腾出内存
    output = clf(X, A)
    loss_test = F.nll_loss(output[idx_test], Y[idx_test])
    acc_test = accuracy(output[idx_test], Y[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

train()
test()
