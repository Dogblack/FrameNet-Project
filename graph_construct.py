import torch
import dgl

from data_process4 import get_fe_tabel, get_frame_tabel,load_data_pd
from relation_extraction import parserelationFiles


def graph_construct(fr_path='', fr_file='frame.csv', fe_path='', fe_file='FE.csv'):
    frame_id_to_label, frame_name_to_label, frame_name_to_id = get_frame_tabel(path=fr_path, file=fr_file)
    fe_id_to_label, fe_name_to_label, fe_name_to_id, fe_id_to_type = get_fe_tabel(path=fe_path, file=fe_file)
    fr_sub, fr_sup, fe_sub, fe_sup, fr_label, fe_label, rel_name, fe_sub_to_frid, fe_sup_to_frid= parserelationFiles()

    data_dic = {}

    # frame to frame
    for item in zip(fr_sup, fr_sub, fr_label):
        #print(item[2])
        data_dic.setdefault(('frame', rel_name[item[2]], 'frame'), [[], []])
        data_dic[('frame', rel_name[item[2]], 'frame')][0].append(frame_id_to_label[item[0]])
        data_dic[('frame', rel_name[item[2]], 'frame')][1].append(frame_id_to_label[item[1]])

    # fe to fe
    data_dic.setdefault(('fe', 'fe_to_fe', 'fe'), [[], []])
    for item in zip(fe_sup, fe_sub, fe_label):
        data_dic[('fe', 'fe_to_fe', 'fe')][0].append(fe_id_to_label[item[0]])
        data_dic[('fe', 'fe_to_fe', 'fe')][1].append(fe_id_to_label[item[1]])
        data_dic[('fe', 'fe_to_fe', 'fe')][0].append(fe_id_to_label[item[1]])
        data_dic[('fe', 'fe_to_fe', 'fe')][1].append(fe_id_to_label[item[0]])

    # frame to fe
    fe_dt = load_data_pd(dataset_path=fe_path, file=fe_file)
    data_dic[('frame', 'fr_to_fe', 'fe')] = [[], []]
    data_dic[('fe', 'fe_to_fr', 'frame')] = [[], []]
    for idx in range(len(fe_dt['FrameID'])):
        data_dic[('frame', 'fr_to_fe', 'fe')][0].append(frame_id_to_label[fe_dt['FrameID'][idx]])
        data_dic[('frame', 'fr_to_fe', 'fe')][1].append(fe_id_to_label[fe_dt['ID'][idx]])

        data_dic[('fe', 'fe_to_fr', 'frame')][0].append(fe_id_to_label[fe_dt['ID'][idx]])
        data_dic[('fe', 'fe_to_fr', 'frame')][1].append(frame_id_to_label[fe_dt['FrameID'][idx]])

    for key in data_dic.keys():
        data_dic[key] = (torch.Tensor(data_dic[key][0]).long(), torch. Tensor(data_dic[key][1]).long())

    g = dgl.heterograph(data_dic)

    return g

if __name__ == '__main__':

    g = graph_construct()
    print(g)
