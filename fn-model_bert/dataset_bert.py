
import torch
import numpy as np

from torch.utils.data import Dataset
from config_bert import get_opt
from data_process_bert import get_frame_tabel, get_fe_tabel, get_fe_list, get_lu_list,DataConfig
from utils import get_mask_from_index

class FrameNetDataset(Dataset):

    def __init__(self, opt, config, data_dic, device):
        super(FrameNetDataset, self).__init__()
        print('begin load data')
        self.data_dic = data_dic
        self.fe_id_to_label, self.fe_name_to_label, self.fe_name_to_id, self.fe_id_to_type = get_fe_tabel('parsed-v1.5/', 'FE.csv')
        self.frame_id_to_label, self.frame_name_to_label, self.frame_name_to_id = get_frame_tabel('parsed-v1.5/', 'frame.csv')

        # self.word_index = config.word_index
        # self.lemma_index = config.lemma_index
        # self.pos_index = config.pos_index
        # self.rel_index = config.rel_index

        self.fe_num = len(self.fe_id_to_label)
        self.frame_num = len(self.frame_id_to_label)
        self.batch_size = opt.batch_size
        print(self.fe_num)
        print(self.frame_num)
        self.dataset_len = len(self.data_dic)

        self.fe_mask_list = get_fe_list('parsed-v1.5/', self.fe_num, self.fe_id_to_label)
        self.lu_list, self.lu_id_to_name,\
        self.lu_name_to_id = get_lu_list('parsed-v1.5/',
                                          self.frame_num, self.fe_num,
                                          self.frame_id_to_label,
                                          self.fe_mask_list)

        self.word_ids = []
        # self.lemma_ids = []
        # self.pos_ids = []
        # self.head_ids = []
        # self.rel_ids = []

        self.lengths = []
        self.mask = []
        self.target_head = []
        self.target_tail = []
        self.target_type = []
        self.fe_head = []
        self.fe_tail = []
        self.fe_type = []
        self.fe_coretype = []
        self.sent_length = []
        self.fe_cnt = []
        self.fe_cnt_with_padding =[]
        self.fe_mask = []
        self.lu_mask = []
        self.token_type_ids = []
        self.target_mask_ids = []

        self.device = device
        self.oov_frame = 0
        self.long_span = 0
        self.error_span = 0
        self.fe_coretype_table = {}
        self.target_mask = {}

        for idx in self.fe_id_to_type.keys():
            if self.fe_id_to_type[idx] == 'Core':
                self.fe_coretype_table[self.fe_id_to_label[idx]] = 1
            else:
                self.fe_coretype_table[self.fe_id_to_label[idx]] = 0



        for key in self.data_dic.keys():
            self.build_target_mask(key,opt.maxlen)


        for key in self.data_dic.keys():
            self.pre_process(key, opt)

        self.pad_dic_cnt = self.dataset_len % opt.batch_size


        for idx,key in enumerate(self.data_dic.keys()):
            if idx >= self.pad_dic_cnt:
                break
            self.pre_process(key, opt,filter=False)

        self.dataset_len+=self.pad_dic_cnt

        print('load data finish')
        print('oov frame = ', self.oov_frame)
        print('long_span = ', self.long_span)
        print('dataset_len = ', self.dataset_len)

    def __len__(self):
        self.dataset_len = int(self.dataset_len / self.batch_size) * self.batch_size
        return self.dataset_len

    def __getitem__(self, item):
        word_ids = torch.Tensor(self.word_ids[item]).long().to(self.device)
        # lemma_ids = torch.Tensor(self.lemma_ids[item]).long().to(self.device)
        # pos_ids = torch.Tensor(self.pos_ids[item]).long().to(self.device)
        # head_ids = torch.Tensor(self.head_ids[item]).long().to(self.device)
        # rel_ids = torch.Tensor(self.rel_ids[item]).long().to(self.device)
        lengths = torch.Tensor([self.lengths[item]]).long().to(self.device)
        mask = torch.Tensor(self.mask[item]).long().to(self.device)
        target_head = torch.Tensor([self.target_head[item]]).long().to(self.device)
        target_tail = torch.Tensor([self.target_tail[item]]).long().to(self.device)
        target_type = torch.Tensor([self.target_type[item]]).long().to(self.device)
        fe_head = torch.Tensor(self.fe_head[item]).long().to(self.device)
        fe_tail = torch.Tensor(self.fe_tail[item]).long().to(self.device)
        fe_type = torch.Tensor(self.fe_type[item]).long().to(self.device)
        fe_cnt = torch.Tensor([self.fe_cnt[item]]).long().to(self.device)
        fe_cnt_with_padding = torch.Tensor([self.fe_cnt_with_padding[item]]).long().to(self.device)
        fe_mask = torch.Tensor(self.fe_mask[item]).long().to(self.device)
        lu_mask = torch.Tensor(self.lu_mask[item]).long().to(self.device)
        token_type_ids = torch.Tensor(self.token_type_ids[item]).long().to(self.device)
        sent_length = torch.Tensor([self.sent_length[item]]).long().to(self.device)
        target_mask_ids = torch.Tensor(self.target_mask_ids[item]).long().to(self.device)
        # print(fe_cnt)


        return (word_ids, lengths, mask, target_head, target_tail, target_type,
                fe_head, fe_tail, fe_type, fe_cnt, fe_cnt_with_padding,
                fe_mask, lu_mask, token_type_ids,sent_length,target_mask_ids)

    def pre_process(self, key, opt,filter=True):
        if self.data_dic[key]['target_type'] not in self.frame_name_to_label:
            self.oov_frame += 1
            self.dataset_len -= 1
            return

        target_id = self.frame_name_to_id[self.data_dic[key]['target_type']]
        if filter:
            self.long_span += self.remove_error_span(key, self.data_dic[key]['span_start'],
                                                 self.data_dic[key]['span_end'], self.data_dic[key]['span_type'], target_id, 20)

        # word_ids = [self.word_index[word] for word in self.data_dic[key]['word_list']]
        # lemma_ids = [self.lemma_index[lemma] for lemma in self.data_dic[key]['lemma_list']]
        # pos_ids = [self.pos_index[pos] for pos in self.data_dic[key]['pos_list']]
        # head_ids = [self.word_index[head] for head in self.data_dic[key]['dep_list'][0]]
        # rel_ids = [self.rel_index[rel] for rel in self.data_dic[key]['dep_list'][1]]

        self.word_ids.append(self.data_dic[key]['tokenized_ids'])
        # self.lemma_ids.append(lemma_ids)
        # self.pos_ids.append(pos_ids)
        # self.head_ids.append(head_ids)
        # self.rel_ids.append(rel_ids)
        self.lengths.append(self.data_dic[key]['length'])

        self.mask.append(self.data_dic[key]['attention_mask'])
        self.target_head.append(self.data_dic[key]['target_idx'][0])
        self.target_tail.append(self.data_dic[key]['target_idx'][1])

        # mask = get_mask_from_index(torch.Tensor([int(self.data_dic[key]['length'])]), opt.maxlen).squeeze()
        # self.mask.append(mask)

        token_type_ids = build_token_type_ids(self.data_dic[key]['target_idx'][0], self.data_dic[key]['target_idx'][1], opt.maxlen)
        # token_type_ids +=self.target_mask[key[0]]
        self.token_type_ids.append(token_type_ids)
        self.target_mask_ids.append(self.target_mask[key[0]])

        self.target_type.append(self.frame_name_to_label[self.data_dic[key]['target_type']])

        # print(self.frame_tabel[self.fe_data[key]['frame_ID']])

        if self.data_dic[key]['length'] <= opt.maxlen:
            sent_length = self.data_dic[key]['length']
        else:
            sent_length = opt.maxlen
        self.sent_length.append(sent_length)

        lu_name = self.data_dic[key]['lu']
        self.lu_mask.append(self.lu_list[lu_name]['lu_mask'])
        self.fe_mask.append(self.lu_list[lu_name]['fe_mask'])

        fe_head = self.data_dic[key]['span_start']
        fe_tail = self.data_dic[key]['span_end']



        while len(fe_head) < opt.fe_padding_num:
            fe_head.append(min(sent_length-1, opt.maxlen-1))

        while len(fe_tail) < opt.fe_padding_num:
            fe_tail.append(min(sent_length-1,opt.maxlen-1))

        self.fe_head.append(fe_head[0:opt.fe_padding_num])
        self.fe_tail.append(fe_tail[0:opt.fe_padding_num])

        fe_type = [self.fe_name_to_label[(item, target_id)] for item in self.data_dic[key]['span_type']]

        self.fe_cnt.append(min(len(fe_type), opt.fe_padding_num))
        self.fe_cnt_with_padding.append(min(len(fe_type)+1, opt.fe_padding_num))

        while len(fe_type) < opt.fe_padding_num:
            fe_type.append(self.fe_num)
            # fe_coretype.append('0')

        self.fe_type.append(fe_type[0:opt.fe_padding_num])

    def remove_error_span(self, key, fe_head_list, fe_tail_list, fe_type_list, target_id, span_maxlen):
        indices = []
        for index in range(len(fe_head_list)):
            if fe_tail_list[index] - fe_head_list[index] >= span_maxlen:
                indices.append(index)
            elif fe_tail_list[index] < fe_head_list[index]:
                indices.append(index)


            elif (fe_type_list[index], target_id) not in self.fe_name_to_label:
                indices.append(index)

            else:
                for i in range(index):
                    if i not in indices:
                        if fe_head_list[index] >= fe_head_list[i] and fe_head_list[index] <= fe_tail_list[i]:
                            indices.append(index)
                            break

                        elif fe_tail_list[index] >= fe_head_list[i] and fe_tail_list[index] <= fe_tail_list[i]:
                            indices.append(index)
                            break
                        elif fe_tail_list[index] <= fe_head_list[i] and fe_tail_list[index] >= fe_tail_list[i]:
                            indices.append(index)
                            break
                        else:
                            continue

        fe_head_list_filter = [i for j, i in enumerate(fe_head_list) if j not in indices]
        fe_tail_list_filter = [i for j, i in enumerate(fe_tail_list) if j not in indices]
        fe_type_list_filter = [i for j, i in enumerate(fe_type_list) if j not in indices]
        self.data_dic[key]['span_start'] = fe_head_list_filter
        self.data_dic[key]['span_end'] = fe_tail_list_filter
        self.data_dic[key]['span_type'] = fe_type_list_filter

        return len(indices)

    def build_target_mask(self,key,maxlen):
        self.target_mask.setdefault(key[0], [0]*maxlen)

        target_head = self.data_dic[key]['target_idx'][0]
        target_tail = self.data_dic[key]['target_idx'][1]
        self.target_mask[key[0]][target_head] = 1
        self.target_mask[key[0]][target_tail] = 1





def build_token_type_ids(target_head, target_tail, maxlen):
    token_type_ids = [0]*maxlen
    token_type_ids[target_head] = 1
    token_type_ids[target_tail] = 1
    # token_type_ids[target_head:target_tail+1] = [1]*(target_tail+1-target_head)

    return token_type_ids


if __name__ == '__main__':
    opt = get_opt()
    config = DataConfig(opt)
    if torch.cuda.is_available():
        device = torch.device(opt.cuda)
    else:
        device = torch.device('cpu')
    dataset = FrameNetDataset(opt, config, config.test_instance_dic, device)
    print(dataset.error_span)
