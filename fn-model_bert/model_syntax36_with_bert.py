import numpy as np
from typing import List,Tuple
import os
import json

import torch.nn as nn
import torch
import torch.nn.functional as F

from utils import batched_index_select,get_mask_from_index,generate_perm_inv
from pytorch_pretrained_bert import BertTokenizer,BertModel,BertConfig,BertAdam
from config_bert import PLMConfig,get_opt

class Mlp(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mlp, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class Relu_Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Relu_Linear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class CnnNet(nn.Module):
    def __init__(self, kernel_size, seq_length, input_size, output_size):
        super(CnnNet, self).__init__()
        self.seq_length = seq_length
        self.output_size = output_size
        self.kernel_size = kernel_size

        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=self.kernel_size
                              , padding=1)
        self.mp = nn.MaxPool1d(kernel_size=self.seq_length)

    def forward(self, input_emb):
        input_emb = input_emb.permute(0, 2, 1)
        x = self.conv(input_emb)
        output = self.mp(x).squeeze()

        return output


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, attention_type='affine'):
        super(PointerNet, self).__init__()

        assert attention_type in ('affine', 'dot_prod')
        if attention_type == 'affine':
            self.src_encoding_linear = Mlp(src_encoding_size, query_vec_size)

        self.src_linear = Mlp(src_encoding_size,src_encoding_size)
        self.activate = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.fc =nn.Linear(src_encoding_size*2,src_encoding_size, bias=True)

        self.attention_type = attention_type

    def forward(self, src_encodings, src_token_mask,query_vec,head_vec=None):

        # (batch_size, 1, src_sent_len, query_vec_size)
        if self.attention_type == 'affine':
            src_encod = self.src_encoding_linear(src_encodings).unsqueeze(1)
            head_weights = self.src_linear(src_encodings).unsqueeze(1)

        # (batch_size, tgt_action_num, query_vec_size, 1)
        if head_vec is not None:
            src_encod = torch.cat([src_encod,head_weights],dim = -1)
            q = torch.cat([head_vec, query_vec], dim=-1).permute(1, 0, 2).unsqueeze(3)


        else:
            q = query_vec.permute(1, 0, 2).unsqueeze(3)

        weights = torch.matmul(src_encod, q).squeeze(3)
        ptr_weights = weights.permute(1, 0, 2)

        # if head_vec is not None:
        #     src_weights = torch.matmul(head_weights, q_h).squeeze(3)
        #     src_weights = src_weights.permute(1, 0, 2)
        #     ptr_weights = weights+src_weights
        #
        # else:
        #    ptr_weights = weights

        ptr_weights_masked = ptr_weights.clone().detach()
        if src_token_mask is not None:
            # (tgt_action_num, batch_size, src_sent_len)
            src_token_mask=1-src_token_mask.byte()
            src_token_mask = src_token_mask.unsqueeze(0).expand_as(ptr_weights)
            # ptr_weights.data.masked_fill_(src_token_mask, -float('inf'))
            ptr_weights_masked.data.masked_fill_(src_token_mask, -float('inf'))

        # ptr_weights =self.activate(ptr_weights)

        return ptr_weights,ptr_weights_masked


class Encoder(nn.Module):
    def __init__(self, opt, config, bert_frozen=False):
        super(Encoder, self).__init__()
        self.opt =opt
        self.hidden_size = opt.rnn_hidden_size
        self.emb_size = opt.encoder_emb_size
        self.rnn_input_size = self.emb_size+opt.token_type_emb_size
        # self.word_number = config.word_number
        # self.lemma_number = config.lemma_number
        self.maxlen = opt.maxlen

        bert_config = BertConfig.from_json_file(PLMConfig.CONFIG_PATH)
        self.bert = BertModel.from_pretrained(PLMConfig.MODEL_PATH)

        if bert_frozen:
            print('Bert grad is false')
            for param in self.bert.parameters():
                param.requires_grad = False

        self.bert_hidden_size = bert_config.hidden_size
        # self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)


        self.dropout = 0.2
        # self.word_embedding = word_embedding
        # self.lemma_embedding = lemma_embedding
        # self.pos_embedding = nn.Embedding(config.pos_number, opt.pos_emb_size)
        # self.rel_embedding = nn.Embedding(config.rel_number,opt.rel_emb_size)
        self.token_type_embedding = nn.Embedding(3, opt.token_type_emb_size)
        self.cell_name = opt.cell_name

        # self.embedded_linear = nn.Linear(self.emb_size*2+opt.pos_emb_size+opt.token_type_emb_size+opt.sent_emb_size,
        #                                 self.rnn_input_size)
        # self.syntax_embedded_linear = nn.Linear(self.emb_size*2+opt.rel_emb_size+opt.token_type_emb_size,
        #                                  self.rnn_input_size)
        # self.output_combine_linear = nn.Linear(4*self.hidden_size, 2*self.hidden_size)

        self.target_linear = nn.Linear(2*self.hidden_size, 2*self.hidden_size)

        self.relu_linear = Relu_Linear(4*self.hidden_size+self.rnn_input_size, opt.decoder_emb_size)

        if self.cell_name == 'gru':
            self.rnn = nn.GRU(self.rnn_input_size, self.hidden_size,num_layers=self.opt.num_layers,
                              dropout=self.dropout,bidirectional=True, batch_first=True)
            # self.syntax_rnn = nn.GRU(self.rnn_input_size, self.hidden_size,num_layers=self.opt.num_layers,
            #                   dropout=self.dropout,bidirectional=True, batch_first=True)
        elif self.cell_name == 'lstm':
            self.rnn = nn.LSTM(self.rnn_input_size, self.hidden_size,num_layers=self.opt.num_layers,
                               dropout=self.dropout,bidirectional=True, batch_first=True)
            # self.syntax_rnn = nn.LSTM(self.rnn_input_size, self.hidden_size,num_layers=self.opt.num_layers,
            #                    dropout=self.dropout,bidirectional=True, batch_first=True)
        else:
            print('cell_name error')

    def forward(self, word_input: torch.Tensor, lengths:torch.Tensor, frame_idx, token_type_ids=None, attention_mask=None,target_mask_ids=None):
        
        # word_embedded = self.word_embedding(word_input)
        # lemma_embedded = self.lemma_embedding(lemma_input)
        # pos_embedded = self.pos_embedding(pos_input)
        # head_embedded = self.word_embedding(head_input)
        # rel_embedded = self.rel_embedding(rel_input)
        # type_ids =torch.add(token_type_ids, target_mask_ids)
        token_type_embedded = self.token_type_embedding(token_type_ids)
        #print(token_type_embedded)
        # print(token_type_ids.size())
        # print(target_mask_ids.size())
        # print(token_type_embedded.size())
        hidden_state, cls = self.bert(word_input, token_type_ids, \
                                                       attention_mask=attention_mask,\
                                                       output_all_encoded_layers=False)

        embedded = torch.cat([hidden_state.squeeze(),token_type_embedded], dim=-1)
        # sent_embedded = self.cnn(embedded)

        # sent_embedded = sent_embedded.expand([self.opt.maxlen, self.opt.batch_size, self.opt.sent_emb_size]).permute(1, 0, 2)
        #embedded = torch.cat([embedded,sent_embedded], dim=-1)
        #embedded = self.embedded_linear(embedded)

        # syntax embedding
        # syntax_embedded = torch.cat([word_embedded,head_embedded,rel_embedded,token_type_ids],dim=-1)
        # syntax_embedded = self.syntax_embedded_linear(syntax_embedded)

        lengths=lengths.squeeze()
        # sorted before pack
        l = lengths.cpu().numpy()
        perm_idx = np.argsort(-l)
        perm_idx_inv = generate_perm_inv(perm_idx)

        embedded = embedded[perm_idx]
        # syntax_embedded = syntax_embedded[perm_idx]

        if lengths is not None:
            rnn_input = nn.utils.rnn.pack_padded_sequence(embedded, lengths=lengths[perm_idx],
                                                          batch_first=True)
            # syntax_rnn_input =  nn.utils.rnn.pack_padded_sequence(syntax_embedded, lengths=lengths[perm_idx],
            #                                               batch_first=True)
        output, hidden = self.rnn(rnn_input)
        #syntax_output, syntax_hidden = self.rnn(syntax_rnn_input)

        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=self.maxlen, batch_first=True)
            # syntax_output, _ = nn.utils.rnn.pad_packed_sequence(syntax_output, total_length=self.maxlen, batch_first=True)


        # print(output.size())
        # print(hidden.size())

        output = output[perm_idx_inv]
        # syntax_output = syntax_output[perm_idx_inv]

        if self.cell_name == 'gru':
            hidden = hidden[:, perm_idx_inv]
            hidden = (lambda a: sum(a)/(2*self.opt.num_layers))(torch.split(hidden, 1, dim=0))

            # syntax_hidden = syntax_hidden[:, perm_idx_inv]
            # syntax_hidden = (lambda a: sum(a)/(2*self.opt.num_layers))(torch.split(syntax_hidden, 1, dim=0))
            # hidden = (hidden + syntax_hidden) / 2

        elif self.cell_name == 'lstm':
            hn0 = hidden[0][:, perm_idx_inv]
            hn1 = hidden[1][:, perm_idx_inv]
            # sy_hn0 = syntax_hidden[0][:, perm_idx_inv]
            # sy_hn1 = syntax_hidden[1][:, perm_idx_inv]
            hn  = tuple([hn0,hn1])
            hidden = tuple(map(lambda state: sum(torch.split(state, 1, dim=0))/(2*self.opt.num_layers), hn))


        target_state_head = batched_index_select(target=output, indices=frame_idx[0])
        target_state_tail = batched_index_select(target=output, indices=frame_idx[1])
        target_state = (target_state_head + target_state_tail) / 2
        target_state = self.target_linear(target_state)

        target_emb_head = batched_index_select(target=embedded, indices=frame_idx[0])
        target_emb_tail = batched_index_select(target=embedded, indices=frame_idx[1])
        target_emb = (target_emb_head + target_emb_tail) / 2

        attentional_target_state = type_attention(attention_mask=target_mask_ids, hidden_state=output,
                                                   target_state=target_state)


        target_state =torch.cat([target_state.squeeze(),attentional_target_state.squeeze(), target_emb.squeeze()], dim=-1)
        target = self.relu_linear(target_state)

        # print(output.size())
        return output, hidden, target


class Decoder(nn.Module):
    def __init__(self, opt, embedding_frozen=False):
        super(Decoder, self).__init__()

        # rnn _init_
        self.opt = opt
        self.cell_name = opt.cell_name
        self.emb_size = opt.decoder_emb_size
        self.hidden_size = opt.decoder_hidden_size
        self.encoder_hidden_size = opt.rnn_hidden_size

        # decoder _init_
        self.decodelen = opt.fe_padding_num+1
        self.frame_embedding = nn.Embedding(opt.frame_number+1, self.emb_size)
        self.frame_fc_layer =Mlp(self.emb_size, opt.frame_number+1)
        self.role_embedding = nn.Embedding(opt.role_number+1, self.emb_size)
        self.role_feature_layer = nn.Linear(2*self.emb_size, self.emb_size)
        self.role_fc_layer = nn.Linear(self.hidden_size+self.emb_size, opt.role_number+1)

        self.head_fc_layer = Mlp(self.hidden_size+self.emb_size, self.hidden_size)
        self.tail_fc_layer = Mlp(self.hidden_size+self.emb_size, self.hidden_size)

        self.span_fc_layer = Mlp(4 * self.encoder_hidden_size + self.emb_size, self.emb_size)

        self.next_input_fc_layer = Mlp(self.hidden_size+self.emb_size, self.emb_size)
        
        if embedding_frozen is True:
            for param in self.frame_embedding.parameters():
                param.requires_grad = False
            for param in self.role_embedding.parameters():
                param.requires_grad = False

        if self.cell_name == 'gru':
            self.frame_rnn = nn.GRU(self.emb_size, self.hidden_size, batch_first=True)
            self.ent_rnn = nn.GRU(self.emb_size, self.hidden_size, batch_first=True)
            self.role_rnn = nn.GRU(self.emb_size, self.hidden_size, batch_first=True)
        if self.cell_name == 'lstm':
            self.frame_rnn = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.ent_rnn = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.role_rnn = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)

        # pointer _init_
        self.ent_pointer = PointerNet(query_vec_size=self.hidden_size, src_encoding_size=2*self.encoder_hidden_size)
        self.head_pointer = PointerNet(query_vec_size=self.hidden_size, src_encoding_size=2*self.encoder_hidden_size)
        self.tail_pointer = PointerNet(query_vec_size=self.hidden_size, src_encoding_size=2*self.encoder_hidden_size)

    def forward(self, encoder_output: torch.Tensor, encoder_state: torch.Tensor, target_state: torch.Tensor,
                attention_mask: torch.Tensor, fe_mask=None, lu_mask=None):
        pred_frame_list = []
        pred_head_list = []
        pred_tail_list = []
        pred_role_list = []

        pred_frame_action = []

        pred_head_action = []
        pred_tail_action = []
        pred_role_action = []

        frame_decoder_state = encoder_state
        role_decoder_state = encoder_state

        input = target_state
        # print(input.size())

        span_mask = attention_mask.clone()
        for t in range(self.decodelen):
            # frame pred
            output, frame_decoder_state = self.decode_step(self.frame_rnn, input=input,
                                                           decoder_state=frame_decoder_state)

            pred_frame_weight = self.frame_fc_layer(target_state.squeeze())
            pred_frame_weight_masked = pred_frame_weight.clone().detach()

            if lu_mask is not None:
                LU_mask = 1-lu_mask
                pred_frame_weight_masked.data.masked_fill_(LU_mask.byte(), -float('inf'))

            pred_frame_indices = torch.argmax(pred_frame_weight_masked.squeeze(), dim=-1).squeeze()

            pred_frame_list.append(pred_frame_weight)
            pred_frame_action.append(pred_frame_indices)

            frame_emb = self.frame_embedding(pred_frame_indices)

            head_input = self.head_fc_layer(torch.cat([output.squeeze(), frame_emb], dim=-1))
            tail_input = self.tail_fc_layer(torch.cat([output.squeeze(), frame_emb], dim=-1))

            head_pointer_weight, head_pointer_weight_masked = self.head_pointer(src_encodings=encoder_output,
                                                                                src_token_mask=span_mask,
                                                                                query_vec=head_input.view(1, self.opt.batch_size, -1))

            head_indices = torch.argmax(head_pointer_weight_masked.squeeze(), dim=-1).squeeze()
            head_target = batched_index_select(target=encoder_output, indices=head_indices.squeeze())
            head_mask = head_mask_update(span_mask, head_indices=head_indices, max_len=self.opt.maxlen)

            tail_pointer_weight, tail_pointer_weight_masked = self.tail_pointer(src_encodings=encoder_output,
                                                                                src_token_mask=head_mask,
                                                                                query_vec=tail_input.view(1, self.opt.batch_size,-1),
                                                                                head_vec=head_target.view(1,self.opt.batch_size,-1))

            tail_indices = torch.argmax(tail_pointer_weight_masked.squeeze(), dim=-1).squeeze()
            # tail_target = batched_index_select(target=bert_hidden_state,indices=tail_indices.squeeze())

            span_mask = span_mask_update(attention_mask=span_mask, head_indices=head_indices,
                                       tail_indices=tail_indices, max_len=self.opt.maxlen)

            pred_head_list.append(head_pointer_weight)
            pred_tail_list.append(tail_pointer_weight)
            pred_head_action.append(head_indices)
            pred_tail_action.append(tail_indices)

            # role pred

            # print(ent_target.size())
            # print(head_target.size())
            # print(tail_target.size())
            # print(bert_hidden_state.size())
            # print(tail_pointer_weight.size())
            # print(tail_indices.size())
            # print(output.size())

            # next step
            # head_target = batched_index_select(target=bert_hidden_state, indices=head_indices.squeeze())

            tail_target = batched_index_select(target=encoder_output, indices=tail_indices.squeeze())

            # head_context =local_attention(attention_mask=attention_mask, hidden_state=encoder_output,
            #                               frame_idx=(head_indices, tail_indices), target_state=head_target,
            #                               window_size=0, max_len=self.opt.maxlen)
            #
            # tail_context =local_attention(attention_mask=attention_mask, hidden_state=encoder_output,
            #                               frame_idx=(head_indices, tail_indices), target_state=tail_target,
            #                               window_size=0, max_len=self.opt.maxlen)

            span_input = self.span_fc_layer(torch.cat([head_target+tail_target, head_target-tail_target, frame_emb], dim=-1)).unsqueeze(1)

            output,role_decoder_state = self.decode_step(self.role_rnn, input=span_input,
                                                         decoder_state=role_decoder_state)
            role_target = self.role_fc_layer(torch.cat([span_input,output],dim=-1))
            role_target_masked = role_target.squeeze().clone().detach()
            if fe_mask is not None :
                FE_mask = 1-fe_mask
                # print(FE_mask.size())
                role_target_masked.data.masked_fill_(FE_mask.byte(), -float('inf'))

            role_indices = torch.argmax(role_target_masked.squeeze(), dim=-1).squeeze()
            role_emb = self.role_embedding(role_indices)

            pred_role_list.append(role_target)
            pred_role_action.append(role_indices)

            # next step
            next_input =torch.cat([output, role_emb.unsqueeze(1)], dim=-1)
            input = self.next_input_fc_layer(next_input)

            
            #return
        return pred_frame_list, pred_head_list, pred_tail_list, pred_role_list, pred_frame_action,\
                        pred_head_action, pred_tail_action, pred_role_action

    def decode_step(self, rnn_cell: nn.modules, input: torch.Tensor, decoder_state: torch.Tensor):

        output, state = rnn_cell(input.view(-1, 1, self.emb_size), decoder_state)

        return output, state


class Model(nn.Module):
    def __init__(self, opt, config):
        super(Model, self).__init__()
        # self.word_vectors = config.word_vectors
        # self.lemma_vectors = config.lemma_vectors
        # self.word_embedding = nn.Embedding(config.word_number+1, opt.encoder_emb_size)
        # self.lemma_embedding = nn.Embedding(config.lemma_number+1, opt.encoder_emb_size)
        #
        # if load_emb:
        #     self.load_pretrain_emb()

        self.encoder = Encoder(opt, config)
        self.decoder = Decoder(opt)

    # def load_pretrain_emb(self):
    #     self.word_embedding.weight.data.copy_(torch.from_numpy(self.word_vectors))
    #     self.lemma_embedding.weight.data.copy_(torch.from_numpy(self.lemma_vectors))

    def forward(self, word_ids,  lengths, frame_idx, fe_mask=None, lu_mask=None,
                frame_len=None, token_type_ids=None, attention_mask=None,target_mask_ids=None):
        encoder_output, encoder_state, target_state = self.encoder(word_input=word_ids,  lengths=lengths,
                                                                   frame_idx=frame_idx,
                                                                   token_type_ids=token_type_ids,
                                                                   attention_mask=attention_mask,
                                                                   target_mask_ids=target_mask_ids)

        pred_frame_list, pred_head_list, pred_tail_list, pred_role_list, pred_frame_action, \
        pred_head_action, pred_tail_action, pred_role_action = self.decoder(encoder_output=encoder_output,
                                                                            encoder_state=encoder_state,
                                                                            target_state=target_state,
                                                                            attention_mask=attention_mask,
                                                                            fe_mask=fe_mask, lu_mask=lu_mask)

        # return pred_frame_list, pred_ent_list, pred_head_list, pred_tail_list, pred_role_list, pred_frame_action, \
        # pred_ent_action, pred_head_action, pred_tail_action, pred_role_action

        return {
            'pred_frame_list' : pred_frame_list,
            'pred_head_list' : pred_head_list,
            'pred_tail_list' : pred_tail_list,
            'pred_role_list' :  pred_role_list,
            'pred_frame_action' : pred_frame_action,
            'pred_head_action' : pred_head_action,
            'pred_tail_action' : pred_tail_action,
            'pred_role_action' : pred_role_action
        }


def head_mask_update(attention_mask: torch.Tensor, head_indices: torch.Tensor, max_len):
    indices=head_indices
    indices_mask=1-get_mask_from_index(indices, max_len)
    mask = torch.mul(attention_mask, indices_mask.long())

    return mask


def span_mask_update(attention_mask: torch.Tensor, head_indices: torch.Tensor, tail_indices: torch.Tensor, max_len):
    tail = tail_indices + 1
    head_indices_mask = get_mask_from_index(head_indices, max_len)
    tail_indices_mask = get_mask_from_index(tail, max_len)
    span_indices_mask = tail_indices_mask - head_indices_mask
    span_indices_mask = 1 - span_indices_mask
    mask = torch.mul(attention_mask, span_indices_mask.long())

    return mask


def local_attention(attention_mask: torch.Tensor, hidden_state: torch.Tensor, frame_idx,
                    target_state: torch.Tensor, window_size: int, max_len):

    q = target_state.squeeze().unsqueeze(2)
    context_att = torch.bmm(hidden_state, q).squeeze()
    head = frame_idx[0]-window_size
    tail = frame_idx[1]+window_size
    mask = span_mask_update(attention_mask=attention_mask, head_indices=head.squeeze(),
                            tail_indices=tail.squeeze(), max_len=max_len)
    context_att = context_att.masked_fill_(mask.byte(), -float('inf'))
    context_att = F.softmax(context_att, dim=-1)
    attentional_hidden_state = torch.bmm(hidden_state.permute(0, 2, 1), context_att.unsqueeze(2)).squeeze()

    return attentional_hidden_state

def type_attention(attention_mask: torch.Tensor, hidden_state: torch.Tensor,
                    target_state: torch.Tensor):

    q = target_state.squeeze().unsqueeze(2)
    context_att = torch.bmm(hidden_state, q).squeeze()

    mask = 1-attention_mask

    context_att = context_att.masked_fill_(mask.byte(), -float('inf'))
    context_att = F.softmax(context_att, dim=-1)
    attentional_hidden_state = torch.bmm(hidden_state.permute(0, 2, 1), context_att.unsqueeze(2)).squeeze()

    return attentional_hidden_state


