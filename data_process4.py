import numpy as np
import pandas as pd
import torch

from config import get_opt
from utils import get_mask_from_index
from nltk.parse.stanford import StanfordDependencyParser

def load_data(path):
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    return lines


def instance_process(lines, maxlen):
    instance_dic = {}

    parser = StanfordDependencyParser(r".\stanford-parser-full-2015-12-09\stanford-parser.jar",
                                          r".\stanford-parser-full-2015-12-09\stanford-parser-3.6.0-models.jar"
                                          )
    cnt = 0
    find = False
    word_list_total = []
    for line in lines:
        if line[0:3] == '# i':
            word_list = []
            lemma_list = []
            pos_list = []
            target_idx = [-1, -1]
            span_start = []
            span_end = []
            span_type = []
            length = 0

        elif line[0:3] == '# e':
            instance_dic.setdefault((sent_id, target_type, cnt), {})
            instance_dic[(sent_id, target_type, cnt)]['dep_list'] = dep_parsing(word_list, maxlen, parser)
            instance_dic[(sent_id, target_type, cnt)]['word_list'] = padding_sentence(word_list, maxlen)
            instance_dic[(sent_id, target_type, cnt)]['lemma_list'] = padding_sentence(lemma_list, maxlen)
            instance_dic[(sent_id, target_type, cnt)]['pos_list'] = padding_sentence(pos_list, maxlen)
            instance_dic[(sent_id, target_type, cnt)]['sent_id'] = sent_id

            word_list_total.append(word_list)
            # add 'eos'
            instance_dic[(sent_id, target_type, cnt)]['length'] = int(length)+1
            # instance_dic[(sent_id, target_type, cnt)]['attention_mask'] = get_mask_from_index(sequence_lengths=torch.Tensor([int(length)+1]), max_length=maxlen).squeeze()

            instance_dic[(sent_id, target_type, cnt)]['target_type'] = target_type
            instance_dic[(sent_id, target_type, cnt)]['lu'] = lu
            instance_dic[(sent_id, target_type, cnt)]['target_idx'] = target_idx

            instance_dic[(sent_id, target_type, cnt)]['span_start'] = span_start
            instance_dic[(sent_id, target_type, cnt)]['span_end'] = span_end
            instance_dic[(sent_id, target_type, cnt)]['span_type'] = span_type
            print(cnt)
            cnt += 1
        elif line == '\n':
            continue

        else:
            data_list = line.split('\t')
            word_list.append(data_list[1])
            lemma_list.append(data_list[3])
            pos_list.append(data_list[5])
            sent_id = data_list[6]
            length = data_list[0]

            if data_list[12] != '_' and data_list[13] != '_':
                lu = data_list[12]

                target_type = data_list[13]
                if target_idx == [-1, -1]:
                    target_idx = [int(data_list[0])-1, int(data_list[0])-1]
                else:
                    target_idx[1] =int(data_list[0]) - 1

            if data_list[14] != '_':

                fe = data_list[14].split('-')

                if fe[0] == 'B' and find is False:
                    span_start.append(int(data_list[0]) - 1)
                    find = True

                elif fe[0] == 'O':
                    span_end.append(int(data_list[0]) - 1)
                    span_type.append(fe[-1].replace('\n', ''))
                    find = False

                elif fe[0] == 'S':
                    span_start.append(int(data_list[0]) - 1)
                    span_end.append(int(data_list[0]) - 1)
                    span_type.append(fe[-1].replace('\n', ''))

    return instance_dic

def dep_parsing(word_list: list,maxlen: list,parser):
    res = list(parser.parse(word_list))
    sent = res[0].to_conll(4).split('\n')[:-1]
    #['the', 'DT', '4', 'det']
    line = [line.split('\t') for line in sent]
    head_list = []
    rel_list = []

    distance = 0

    #alignment
    for index in range(len(word_list)-1):
        #end stopwords
        if index-distance >= len(line):
            head_list.append('#')
            rel_list.append('#')
            distance+=1

        elif word_list[index]!=line[index-distance][0]:
            head_list.append('#')
            rel_list.append('#')
            distance+=1
        else:
            rel_list.append(line[index-distance][3])

            if line[index-distance][3] != 'root':
                head_list.append(word_list[int(line[index-distance][2]) - 1])
            else:
                head_list.append(word_list[index])

    head_list.append('eos')
    rel_list.append('eos')

    while len(head_list) < maxlen:
        head_list.append('0')
        rel_list.append('0')

    return (head_list,rel_list)


def padding_sentence(sentence: list,maxlen: int):
    sentence.append('eos')
    while len(sentence) < maxlen:
        sentence.append('0')

    return sentence

class DataConfig:
    def __init__(self,opt):
        exemplar_lines = load_data('fn1.5/conll/exemplar')
        train_lines = load_data('fn1.5/conll/train')
        dev_lines = load_data('fn1.5/conll/dev')
        test_lines = load_data('fn1.5/conll/test')


        self.emb_file_path = opt.emb_file_path
        self.maxlen = opt.maxlen

        if opt.load_instance_dic:
            self.exemplar_instance_dic = np.load(opt.exemplar_instance_path, allow_pickle=True).item()
            self.train_instance_dic = np.load(opt.train_instance_path, allow_pickle=True).item()
            self.dev_instance_dic = np.load(opt.dev_instance_path, allow_pickle=True).item()
            self.test_instance_dic = np.load(opt.test_instance_path, allow_pickle=True).item()

        else:
            print('begin parsing')
            self.exemplar_instance_dic = instance_process(lines=exemplar_lines,maxlen=self.maxlen)
            print('exemplar_instance_dic finish')
            self.train_instance_dic = instance_process(lines=train_lines,maxlen=self.maxlen)
            np.save('train_instance_dic', self.train_instance_dic)
            print('train_instance_dic finish')
            self.dev_instance_dic = instance_process(lines=dev_lines,maxlen=self.maxlen)
            np.save('dev_instance_dic', self.dev_instance_dic)
            print('dev_instance_dic finish')
            self.test_instance_dic = instance_process(lines=test_lines,maxlen=self.maxlen)
            np.save('test_instance_dic', self.test_instance_dic)
            print('test_instance_dic finish')


        self.word_index = {}
        self.lemma_index = {}
        self.pos_index = {}
        self.rel_index = {}

        self.word_number = 0
        self.lemma_number = 0
        self.pos_number = 0
        self.rel_number = 0

        self.build_word_index(self.exemplar_instance_dic)
        self.build_word_index(self.train_instance_dic)
        self.build_word_index(self.dev_instance_dic)
        self.build_word_index(self.test_instance_dic)

        # add # for parsing sign
        self.word_index['#']=self.word_number
        self.word_number+=1

        self.emb_index = self.build_emb_index(self.emb_file_path)

        self.word_vectors = self.get_embedding_weight(self.emb_index, self.word_index, self.word_number)
        self.lemma_vectors = self.get_embedding_weight(self.emb_index, self.lemma_index, self.lemma_number)

    def build_word_index(self, dic):
        for key in dic.keys():
            word_list =dic[key]['word_list']
            lemma_list = dic[key]['lemma_list']
            pos_list = dic[key]['pos_list']
            rel_list = dic[key]['dep_list'][1]

            # print(row)
            for word in word_list:
                if word not in self.word_index.keys():
                    self.word_index[word]=self.word_number
                    self.word_number += 1

            for lemma in lemma_list:
                if lemma not in self.lemma_index.keys():
                    self.lemma_index[lemma]=self.lemma_number
                    self.lemma_number += 1

            for pos in pos_list:
                if pos not in self.pos_index.keys():
                    self.pos_index[pos] = self.pos_number
                    self.pos_number += 1

            for rel in rel_list:
                if rel not in self.rel_index.keys():
                    self.rel_index[rel] = self.rel_number
                    self.rel_number += 1

    def build_emb_index(self, file_path):
        data = open(file_path, 'r', encoding='utf-8')
        emb_index = {}
        for items in data:
            item = items.split()
            word = item[0]
            weight = np.asarray(item[1:], dtype='float32')
            emb_index[word] = weight

        return emb_index

    def get_embedding_weight(self,embed_dict, words_dict, words_count, dim=200):

        exact_count = 0
        fuzzy_count = 0
        oov_count = 0
        print("loading pre_train embedding by avg for out of vocabulary.")
        embeddings = np.zeros((int(words_count) + 1, int(dim)))
        inword_list = {}
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = embed_dict[word]
                inword_list[words_dict[word]] = 1
                # 准确匹配
                exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = embed_dict[word.lower()]
                inword_list[words_dict[word]] = 1
                # 模糊匹配
                fuzzy_count += 1
            else:
                # 未登录词
                oov_count += 1
                # print(word)
        # 对已经找到的词向量平均化
        sum_col = np.sum(embeddings, axis=0) / len(inword_list)  # avg
        sum_col /= np.std(sum_col)
        for i in range(words_count):
            if i not in inword_list:
                embeddings[i] = sum_col

        embeddings[int(words_count)] = [0] * dim
        final_embed = np.array(embeddings)
        # print('exact_count: ',exact_count)
        # print('fuzzy_count: ', fuzzy_count)
        # print('oov_count: ', oov_count)
        return final_embed


def load_data_pd(dataset_path,file):
    # df=csv.reader(open(dataset_path+file,encoding='utf-8'))
    # df = json.load(open(file_path,encoding='utf-8'))
    df=pd.read_csv(dataset_path+file, header=0, encoding='utf-8')
    return df


def get_frame_tabel(path, file):
    data = load_data_pd(path, file)

    frame_id_to_label = {}
    frame_name_to_label = {}
    frame_name_to_id = {}
    data_index = 0
    for idx in range(len(data['ID'])):
        if data['ID'][idx] not in frame_id_to_label:
            frame_id_to_label[data['ID'][idx]] = data_index
            frame_name_to_label[data['Name'][idx]] = data_index
            frame_name_to_id[data['Name'][idx]] = data['ID'][idx]

            data_index += 1

    return frame_id_to_label, frame_name_to_label, frame_name_to_id


def get_fe_tabel(path, file):
    data = load_data_pd(path, file)

    fe_id_to_label = {}
    fe_name_to_label = {}
    fe_name_to_id = {}
    fe_id_to_type = {}

    data_index = 0
    for idx in range(len(data['ID'])):
        if data['ID'][idx] not in fe_id_to_label:
            fe_id_to_label[data['ID'][idx]] = data_index
            fe_name_to_label[(data['Name'][idx], data['FrameID'][idx])] = data_index
            fe_name_to_id[(data['Name'][idx], data['FrameID'][idx])] = data['ID'][idx]
            fe_id_to_type[data['ID'][idx]] = data['CoreType'][idx]

            data_index += 1

    return fe_id_to_label, fe_name_to_label, fe_name_to_id, fe_id_to_type


def get_fe_list(path, fe_num, fe_table, file='FE.csv'):
    fe_dt = load_data_pd(path, file)
    fe_mask_list = {}

    print('begin get fe list')
    for idx in range(len(fe_dt['FrameID'])):
        fe_mask_list.setdefault(fe_dt['FrameID'][idx], [0]*(fe_num+1))
        # fe_mask_list[fe_dt['FrameID'][idx]].setdefault('fe_mask', [0]*(fe_num+1))
        fe_mask_list[fe_dt['FrameID'][idx]][fe_table[fe_dt['ID'][idx]]] = 1

    # for key in fe_list.keys():
    #     fe_list[key]['fe_mask'][fe_num] = 1

    return fe_mask_list


def get_lu_list(path, lu_num, fe_num, frame_id_to_label, fe_mask_list, file='LU.csv'):
    lu_dt = load_data_pd(path, file)
    lu_list = {}
    lu_id_to_name = {}
    lu_name_to_id = {}
    #lu_name_to_felist = {}

    for idx in range(len(lu_dt['ID'])):
        lu_name = lu_dt['Name'][idx]
        lu_list.setdefault(lu_name, {})

        lu_list[lu_name].setdefault('fe_mask', [0]*(fe_num+1))
        lu_list[lu_name]['fe_mask'] = list(map(lambda x: x[0]+x[1], zip(lu_list[lu_name]['fe_mask'],
                                           fe_mask_list[lu_dt['FrameID'][idx]])))

        lu_list[lu_name].setdefault('lu_mask', [0]*(lu_num+1))
        lu_list[lu_name]['lu_mask'][frame_id_to_label[lu_dt['FrameID'][idx]]] = 1

        lu_id_to_name[lu_dt['ID'][idx]] = lu_name
        lu_name_to_id[(lu_name, lu_dt['FrameID'][idx])] = lu_dt['ID'][idx]

    for key in lu_list.keys():
        # lu_list[key]['lu_mask'][lu_num] = 1
        lu_list[key]['fe_mask'][fe_num] = 1

    return lu_list, lu_id_to_name, lu_name_to_id


if __name__ == '__main__':
    opt = get_opt()
    config = DataConfig(opt)
    print(config.word_vectors)
    print(config.lemma_number)
    print(config.word_number)
    print(config.pos_number)
    print(config.dep_number)



