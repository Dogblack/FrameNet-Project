import dgl
import dgl.nn as dglnn
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from graph_construct import graph_construct

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, num_frame, num_fe, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats, hid_feats)
                                            for rel in rel_names}, aggregate='mean')
        # self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hid_feats, out_feats)
        #                                     for rel in rel_names}, aggregate='mean')

        self.fr_linear = nn.Linear(num_frame, in_feats)
        self.fe_linear = nn.Linear(num_fe, in_feats)

        self.fr_fc = nn.Linear(hid_feats, num_frame)
        self.fe_fc = nn.Linear(hid_feats, num_fe)

    def forward(self, graph, inputs):

        inputs['frame'] = self.fr_linear(inputs['frame'])
        inputs['fe'] = self.fe_linear(inputs['fe'])
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv1(graph, h)
        h['frame'] = h['frame'] + inputs['frame']
        h['fe'] = h['fe'] + inputs['fe']
        fr_h = self.fr_fc(h['frame'])
        fe_h = self.fe_fc(h['fe'])

        return fr_h, fe_h, h

if __name__ == '__main__':

    frame_graph = graph_construct()
    # 特征初始化
    num_frame = 1019
    num_fe = 9634
    n_features= 200
    save_model_path = './pretrain_model'
    rel_names = ['Causative_of', 'Inchoative_of', 'Inheritance', 'Perspective_on', 'Precedes',
                                'ReFraming_Mapping', 'See_also', 'Subframe', 'Using','fe_to_fe', 'fe_to_fr', 'fr_to_fe']


    frame_graph.nodes['frame'].data['feature'] = torch.eye(num_frame)
    frame_graph.nodes['fe'].data['feature'] = torch.eye(num_fe)
    frame_graph.nodes['frame'].data['label'] = torch.arange(0, num_frame, 1)
    frame_graph.nodes['fe'].data['label'] = torch.arange(0, num_fe, 1)

    model = RGCN(in_feats=n_features, hid_feats=200,
                 num_frame=num_frame, num_fe=num_fe, rel_names=rel_names)

    fr_feats = frame_graph.nodes['frame'].data['feature']
    fe_feats = frame_graph.nodes['fe'].data['feature']
    fr_labels = frame_graph.nodes['frame'].data['label']
    fe_labels = frame_graph.nodes['fe'].data['label']

    #h_dict = model(frame_graph, {'frame': fr_feats, 'fe': fe_feats})

    opt = torch.optim.Adam(model.parameters())

    best_fr_train_acc = 0
    best_fe_train_acc = 0
    loss_list = []
    train_fr_score_list = []
    train_fe_score_list = []

    for epoch in range(200):
        model.train()
        # 输入图和节点特征
        fr_logits, fe_logits, hidden = model(frame_graph, {'frame': fr_feats, 'fe': fe_feats})
        # 计算损失
        loss = F.cross_entropy(fr_logits, fr_labels) + F.cross_entropy(fe_logits, fe_labels)
        # 预测frame
        fr_pred = fr_logits.argmax(1)
        # 计算准确率
        fr_train_acc = (fr_pred == fr_labels).float().mean()
        if best_fr_train_acc < fr_train_acc:
            best_fr_train_acc = fr_train_acc
        train_fr_score_list.append(fr_train_acc)

        # 预测fe
        fe_pred = fe_logits.argmax(1)
        # 计算准确率
        fe_train_acc = (fe_pred == fe_labels).float().mean()
        if best_fe_train_acc < fe_train_acc:
            best_fe_train_acc = fe_train_acc
        train_fe_score_list.append(fe_train_acc)

        # 反向优化
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_list.append(loss.item())
        # 输出训练结果
        print('Loss %.4f, Train fr Acc %.4f (Best %.4f) Train fe Acc %.4f (Best %.4f)' % (
            loss.item(),
            fr_train_acc.item(),
            best_fr_train_acc,
            fe_train_acc.item(),
            best_fe_train_acc
        ))
        #print(frame_graph.nodes['frame'].data['feature'])

    torch.save(model.state_dict(), save_model_path)
    print(hidden)
    torch.save(hidden['frame'], "./frTensor4.pt")
    torch.save(hidden['fe'], "./feTensor4.pt")