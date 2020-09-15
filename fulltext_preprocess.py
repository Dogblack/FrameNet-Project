import numpy as np
import os
import pandas as pd
import csv
import numpy
import torch
import json
import re

from config import get_opt,PLMConfig
from pytorch_pretrained_bert import BertTokenizer
from bert_slot_tokenizer import SlotConverter
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bert_tokenizer=BertTokenizer.from_pretrained(PLMConfig.MODEL_PATH)


def load_data_pd(dataset_path,file):
    #df=csv.reader(open(dataset_path+file,encoding='utf-8'))
    #df = json.load(open(file_path,encoding='utf-8'))
    df=pd.read_csv(dataset_path+file,header=0,encoding='utf-8')
    return df

opt = get_opt()
cnt = 0

def get_sent_dic(path,maxlen,file='sentTable.csv'):
    sent = load_data_pd(path,file)
    #print(sent)
    sent_dic={}
    cnt=0
    print('begin get sent_dic')
    for idx in range(len(sent['ID'])):
        sent_dic.setdefault(sent['ID'][idx],{})
        sent_dic[sent['ID'][idx]]['raw_text']=sent['sentText'][idx]

        padded_sent = '[CLS] '+sent['sentText'][idx]+' [SEP]'
        sent_dic[sent['ID'][idx]]['pad_text'] = padded_sent

        word_list=bert_tokenizer.tokenize(padded_sent)

        #attention_mask
        seq_length = [0] * maxlen
        if len(word_list) < maxlen:
            seq_length[0:len(word_list)]=[1] * len(word_list)
        else:
            seq_length[0:maxlen] = [1] * maxlen

        while len(word_list)< maxlen:
            word_list.append('0')


        sent_dic[sent['ID'][idx]]['tokenized_text'] = word_list
        sent_dic[sent['ID'][idx]]['tokenized_ids'] = bert_tokenizer.convert_tokens_to_ids(word_list)
        sent_dic[sent['ID'][idx]]['attention_mask'] = seq_length

        cnt+=1
        print(cnt)

    print('get sent_dic finish')
    return sent_dic

def get_lu_dic(path):
    lu = load_data_pd(path, 'LU.csv')
    lu_dic = {}
    print('begin get lu_dic')
    cnt=0
    for idx in range(len(lu['ID'])):
        lu_dic.setdefault(lu['ID'][idx], {})
        lu_dic[lu['ID'][idx]]['Name']=lu['Name'][idx].partition('.')[0]
        lu_dic[lu['ID'][idx]]['FrameID'] = lu['FrameID'][idx]
        lu_dic[lu['ID'][idx]]['POS'] = lu['POS'][idx]
        cnt += 1
        print(cnt)

    print('get lu_dic finish')
    return lu_dic


def get_fe_dic(path,sent_dic,lu_dic,file='FETable.csv'):
    fe_anno = load_data_pd(path,file)
    print('begin get fe_dic')
    cnt=0
    error_cnt = 0
    error_list =[]
    #print(sent_dic[692546])
    fe_dic ={}

    raw_sentences={}
    frame_ID ={}
    lu_head_ID={}
    lu_tail_ID={}
    fe_label_ID={}
    fe_head_ID={}
    fe_tail_ID={}
    error_sent_list =[]

    for idx in range(len(fe_anno['frameID'])):
        #fe_sent = fe_anno['SentID'][idx]
        #print(sent_dic[fe_sent])
        fe_sent = fe_anno['ID'][idx]
        if fe_sent in error_list:
            continue
    #
        frame_ID = fe_anno['frameID'][idx]
        fe_dic.setdefault((fe_sent,frame_ID),{})
        fe_dic[(fe_sent,frame_ID)].setdefault('fe_ID',[])
        fe_dic[(fe_sent,frame_ID)].setdefault('fe_cnt',0)
        fe_dic[(fe_sent,frame_ID)]['fe_ID'].append(fe_anno['feID'][idx])
        fe_dic[(fe_sent,frame_ID)]['frame_ID'] = frame_ID
        fe_dic[(fe_sent, frame_ID)]['lu_ID'] = fe_anno['luID'][idx]
        fe_dic[(fe_sent,frame_ID)]['fe_cnt']+=1
    #     sent_dic[fe_sent].setdefault('fe_ID',[])
    #     sent_dic[fe_sent]['fe_ID'].append(fe_anno['FEID'][idx])
    #     sent_dic[fe_sent]['frame_ID']=fe_anno['FrameID'][idx]
    #
        #get lu head and tail
        raw_text=sent_dic[fe_sent]['raw_text']
        tokenized_text=list(sent_dic[fe_sent]['tokenized_text'])

        fe_dic[(fe_sent, frame_ID)]['tokenized_text']=tokenized_text

        lu_start = fe_anno['lu_start'][idx]
        lu_end   = fe_anno['lu_end'][idx]

        if lu_start == -1 or lu_end == -1:
            del fe_dic[(fe_sent,frame_ID)]
            print('error')
            continue

        lu_name =raw_text[lu_start:lu_end+1]

        lu_name_list=bert_tokenizer.tokenize(lu_name)
        print(lu_name_list)
        print(lu_start)
        print(lu_end)

        fe_dic[(fe_sent,frame_ID)]['lu_head']=tokenized_text.index(lu_name_list[0])
        fe_dic[(fe_sent,frame_ID)]['lu_tail'] = tokenized_text.index(lu_name_list[-1])
    #
    #     sent_dic[fe_sent]['lu_head']=tokenized_lu.index(lu_name_list[0])
    #     sent_dic[fe_sent]['lu_tail'] = tokenized_lu.index(lu_name_list[-1])
    #
        # get fe head and tail
        tokenized_lu=list(sent_dic[fe_sent]['tokenized_text'])
        fe_head =fe_anno['start'][idx]
        fe_tail =fe_anno['end'][idx]
        fe_span = raw_text[fe_head:fe_tail+1]
        tokenized_fe = bert_tokenizer.tokenize(fe_span)

    #
        fe_dic[(fe_sent,frame_ID)].setdefault('fe_head',[])
        fe_dic[(fe_sent,frame_ID)].setdefault('fe_tail',[])
        fe_dic[(fe_sent,frame_ID)]['fe_head'].append(tokenized_lu.index(tokenized_fe[0]))
        fe_dic[(fe_sent,frame_ID)]['fe_tail'].append(tokenized_lu.index(tokenized_fe[-1]))
    #
    #     sent_dic[fe_sent].setdefault('fe_head',[])
    #     sent_dic[fe_sent].setdefault('fe_hail',[])
    #     sent_dic[fe_sent]['fe_head'].append(tokenized_lu.index(tokenized_fe[0]))
    #     sent_dic[fe_sent]['fe_hail'].append(tokenized_lu.index(tokenized_fe[-1]))
    #

        #error_sent_list.append(fe_anno['SentID'][idx])
        # if len(error_list)>300:
        #     break
        cnt+=1

        print(cnt)

    return sent_dic,fe_dic,error_list


def get_fe_list(path,fe_num,fe_table,file='FE.csv'):
    fe_dt = load_data_pd(path, file)
    fe_list ={}

    print('begin get fe list')
    for idx in range(len(fe_dt['FrameID'])):
        fe_list.setdefault(fe_dt['FrameID'][idx],{})
        fe_list[fe_dt['frameID'][idx]].setdefault('fe_mask',[0]*(fe_num+1))
        fe_list[fe_dt['frameID'][idx]]['fe_mask'][fe_table[fe_dt['ID'][idx]]]=1

    for key in fe_list.keys():
        fe_list[key]['fe_mask'][fe_num]=1

    return fe_list

def ge_lu_list(path,lu_num,frame_table,file='LU.csv'):
    lu_dt = load_data_pd(path,file)
    lu_list = {}
    luID_list = {}
    for idx in range(len(lu_dt['ID'])):
        lu_name = lu_dt['Name'][idx].partition('.')[0]
        lu_list.setdefault(lu_name,{})
        lu_list[lu_name].setdefault('lu_mask',[0]*(lu_num+1))
        lu_list[lu_name]['lu_mask'][frame_table[lu_dt['FrameID'][idx]]]=1

        luID_list[lu_dt['ID'][idx]]=lu_name

    for key in lu_list.keys():
        lu_list[key]['lu_mask'][lu_num]=1

    return lu_list,luID_list









if __name__=='__main__':
    #根据你数据集所在的位置修改
    dataset_path='./parsed-v1.5/fulltext/dev/'
    #LU.csv所在位置
    lu_path='parsed-v1.5/'




    sent_dic = get_sent_dic(path=dataset_path,maxlen=opt.maxlen)
    #print(sent_dic[1281579])setting

    lu_dic = get_lu_dic(lu_path)

    sent_dic,fe_dic,error_list = get_fe_dic(path=dataset_path,sent_dic=sent_dic,lu_dic=lu_dic)
    print(len(fe_dic))

    #保存成。npy格式，加载时用np.load
    np.save('dev_sent',sent_dic)
    #sent_dic load出来后是一个嵌套dict，key为sentence的ID(sent_id)， sent_dic[ID]={'raw_text','pad_text','tokenized_text',\
    # 'tokenized_ids','attention_mask'}

    #fe_dic load出来后是一个嵌套dict，key为元组（sent_id,frame_id）， fe_dic[(sent_id,frame_id)]={'frame_ID','lu_ID',\
    # 'lu_head','lu_tail','tokenized_text','fe_cnt','fe_ID','fe_head','fe_tail'} 其中 fe_head,fe_tail,fe_ID 为list
    np.save('dev_fe',fe_dic)