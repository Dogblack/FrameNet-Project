import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import multiprocessing as mp

from dataset2 import FrameNetDataset
from torch.utils.data import DataLoader
from utils import get_mask_from_index,seed_everything
from evaluate2 import Eval
from config import get_opt
from data_process2 import DataConfig
from model_syntax import Model


def evaluate(opt, model, dataset, best_metrics=None, show_case=False):
    model.eval()
    print('begin eval')
    evaler = Eval(opt)
    with torch.no_grad():
        test_dl = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=0
        )

        for batch in test_dl:
            word_ids, lemma_ids, pos_ids,head_ids,rel_ids, lengths, attention_mask, target_head, target_tail, \
            target_type, fe_head, fe_tail, fe_type, fe_cnt, \
            fe_cnt_with_padding, fe_mask, lu_mask, token_type_ids = batch

            return_dic = model(word_ids=word_ids, lemma_ids=lemma_ids, pos_ids=pos_ids,head_ids=head_ids,
                               rel_ids=rel_ids,lengths=lengths,
                               frame_idx=(target_head, target_tail),
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask, fe_mask=fe_mask, lu_mask=lu_mask)


            evaler.metrics(batch_size=opt.batch_size, fe_cnt=fe_cnt, gold_fe_type=fe_type, gold_fe_head=fe_head, \
                           gold_fe_tail=fe_tail, gold_frame_type=target_type,
                           pred_fe_type=return_dic['pred_role_action'],
                           pred_fe_head=return_dic['pred_head_action'],
                           pred_fe_tail=return_dic['pred_tail_action'],
                           pred_frame_type=return_dic['pred_frame_action'])


            if show_case:
                print('gold_fe_label = ', fe_type)
                print('pred_fe_label = ', return_dic['pred_role_action'])
                print('gold_head_label = ', fe_head)
                print('pred_head_label = ', return_dic['pred_head_action'])
                print('gold_tail_label = ', fe_tail)
                print('pred_tail_label = ', return_dic['pred_tail_action'])

        metrics = evaler.calculate()


        if best_metrics:

            if metrics[-1] > best_metrics:
                best_metrics = metrics[-1]

                torch.save(model.state_dict(), opt.save_model_path)

            return best_metrics






if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = 1

    # bertconfig = BertConfig()
    # print(bertconfig.CONFIG_PATH)
    mp.set_start_method('spawn')

    opt = get_opt()
    config = DataConfig(opt)

    if torch.cuda.is_available():
        device = torch.device(opt.cuda)
    else:
        device = torch.device('cpu')

    seed_everything(1116)

    epochs = opt.epochs
    model = Model(opt, config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    frame_criterion =nn.CrossEntropyLoss()
    head_criterion = nn.CrossEntropyLoss()
    tail_criterion = nn.CrossEntropyLoss()
    fe_type_criterion = nn.CrossEntropyLoss()

    if os.path.exists(opt.save_model_path) is True:
        model.load_state_dict(torch.load(opt.save_model_path))

    #pretrain_dataset = FrameNetDataset(opt, config, config.exemplar_instance_dic, device)
    train_dataset = FrameNetDataset(opt, config, config.train_instance_dic, device)
    dev_dataset = FrameNetDataset(opt, config, config.dev_instance_dic, device)
    test_dataset = FrameNetDataset(opt, config, config.test_instance_dic, device)

    if opt.mode == 'train':
        for epoch in range(1, epochs):
            train_dl = DataLoader(
                train_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=0
            )
            model.train()
            print('==========epochs= ' + str(epoch))
            step = 0
            sum_loss = 0
            best_metrics = -1
            cnt = 0
            for batch in train_dl:
                optimizer.zero_grad()
                loss = 0
                word_ids, lemma_ids, pos_ids,head_ids, rel_ids, lengths, attention_mask, target_head, target_tail, \
                target_type, fe_head, fe_tail, fe_type, fe_cnt, \
                fe_cnt_with_padding, fe_mask, lu_mask, token_type_ids = batch

                return_dic = model(word_ids=word_ids, lemma_ids=lemma_ids, pos_ids=pos_ids, head_ids=head_ids, rel_ids=rel_ids
                                  ,lengths=lengths,
                                   frame_idx=(target_head, target_tail),
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask, fe_mask=fe_mask, lu_mask=lu_mask)
                # print(return_dic)

                frame_loss = 0
                head_loss = 0
                tail_loss = 0
                type_loss = 0

                for batch_index in range(opt.batch_size):
                    pred_frame_first = return_dic['pred_frame_list'][fe_cnt[batch_index]][batch_index].unsqueeze(0)
                    pred_frame_last = return_dic['pred_frame_list'][0][batch_index].unsqueeze(0)

                    pred_frame_label = pred_frame_last

                    gold_frame_label = target_type[batch_index]
                    # print(gold_frame_label.size())
                    # print(pred_frame_label.size())
                    # print(fe_head)
                    frame_loss += frame_criterion(pred_frame_label, gold_frame_label)


                    for fe_index in range(opt.fe_padding_num):

                        pred_type_label = return_dic['pred_role_list'][fe_index].squeeze()
                        pred_type_label = pred_type_label[batch_index].unsqueeze(0)

                        gold_type_label = fe_type[batch_index][fe_index].unsqueeze(0)
                        type_loss += fe_type_criterion(pred_type_label, gold_type_label)

                        if fe_index >= fe_cnt[batch_index]:
                            break

                        # print(fe_cnt[batch_index])

                        pred_head_label = return_dic['pred_head_list'][fe_index].squeeze()
                        pred_head_label = pred_head_label[batch_index].unsqueeze(0)

                        gold_head_label = fe_head[batch_index][fe_index].unsqueeze(0)
                        #    print(gold_head_label.size())
                        #    print(pred_head_label.size())
                        head_loss += head_criterion(pred_head_label, gold_head_label)

                        pred_tail_label = return_dic['pred_tail_list'][fe_index].squeeze()
                        pred_tail_label = pred_tail_label[batch_index].unsqueeze(0)

                        gold_tail_label = fe_tail[batch_index][fe_index].unsqueeze(0)
                        tail_loss += tail_criterion(pred_tail_label, gold_tail_label)



                    # print(fe_cnt[batch_index])
                    # head_loss /= int(fe_cnt[batch_index])
                    # tail_loss /= int(fe_cnt[batch_index])
                    # type_loss /= int(fe_cnt[batch_index]+1)
                    #
                    # head_loss_total+=head_loss
                    # tail_loss_total+=tail_loss
                    # type_loss_total+=type_loss

                loss = (0.1 * frame_loss + 0.3 * type_loss + 0.3 * head_loss + 0.3 * tail_loss) / (opt.batch_size)
                # loss = (0.3 * head_loss + 0.3 * tail_loss) / (opt.b0.3 * atch_size)
                loss.backward()
                optimizer.step()
                # loss+=frame_loss()
                step += 1
                if step % 20 == 0:
                    print(" | batch loss: %.6f step = %d" % (loss.item(), step))
                # print('gold_frame_label = ',target_type)
                # print('pred_frame_label = ',return_dic['pred_frame_action'])

                for index in range(len(target_type)):
                    if target_type[index] == return_dic['pred_frame_action'][0][index]:
                        cnt += 1
                # print('gold_fe_label = ', fe_type)
                # print('pred_fe_label = ', return_dic['pred_role_action'])
                # print('gold_head_label = ', fe_head)
                # print('pred_head_label = ', return_dic['pred_head_action'])
                # print('gold_tail_label = ', fe_tail)
                # print('pred_tail_label = ', return_dic['pred_tail_action'])
                sum_loss += loss.item()

            print('| epoch %d  avg loss = %.6f' % (epoch, sum_loss / step))
            print('| epoch %d  prec = %.6f' % (epoch, cnt / (opt.batch_size * step)))

            best_metrics=evaluate(opt,model,dev_dataset,best_metrics)


    else:
        evaluate(opt,model,test_dataset,show_case=True)




