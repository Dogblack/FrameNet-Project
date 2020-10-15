import torch

class Eval():
    def __init__(self,opt):
        self.fe_TP = 0
        self.fe_TP_FP = 0
        self.fe_TP_FN = 0

        self.frame_cnt = 0
        self.frame_acc = 0

        self.opt=opt
    def metrics(self,batch_size:int,fe_cnt:torch.Tensor,\
                gold_fe_type:torch.Tensor,gold_fe_head:torch.Tensor,\
                gold_fe_tail:torch.Tensor,gold_frame_type:torch.Tensor,\
                pred_fe_type:torch.Tensor,pred_fe_head:torch.Tensor,\
                pred_fe_tail:torch.Tensor,pred_frame_type:torch.Tensor,
                ):

        self.frame_cnt+=batch_size
        for batch_index in range(batch_size):
            #caculate frame acc
            # print(gold_frame_type[batch_index])
            # print(pred_frame_type[batch_index])
            if gold_frame_type[batch_index] == pred_frame_type[0][batch_index]:
                self.frame_acc += 1

            #update tp_fn
            self.fe_TP_FN+=int(fe_cnt[batch_index])
            # preprocess error
            for idx in range(fe_cnt[batch_index]):
                if gold_fe_head[batch_index][idx] > gold_fe_tail[batch_index][idx]:
                    self.fe_TP_FN -= 1


            #gold_fe_list = gold_fe_type.cpu().numpy().tolist()
            gold_tail_list =gold_fe_tail.cpu().numpy().tolist()
            #update fe_tp and fe_TP_FP
            for fe_index in range(self.opt.fe_padding_num):
                if pred_fe_type[fe_index][batch_index] == self.opt.role_number:
                    break


                #update fe_tp
                if pred_fe_tail[fe_index][batch_index] in gold_tail_list[batch_index]:
                    idx = gold_tail_list[batch_index].index(pred_fe_tail[fe_index][batch_index])
                    

                    if pred_fe_head[fe_index][batch_index]==gold_fe_head[batch_index][idx] and \
                            pred_fe_type[fe_index][batch_index] == gold_fe_type[batch_index][idx]:
                        self.fe_TP+=1

                #update fe_tp_fp
                self.fe_TP_FP+=1

    def calculate(self):
        frame_acc = self.frame_acc / self.frame_cnt
        fe_prec = self.fe_TP / (self.fe_TP_FP+0.000001)
        fe_recall = float(self.fe_TP / self.fe_TP_FN)
        fe_f1 = 2*fe_prec*fe_recall/(fe_prec+fe_recall+0.0000001)

        full_TP = self.frame_acc+self.fe_TP
        full_TP_FP = self.frame_cnt + self.fe_TP_FP
        full_TP_FN = self.frame_cnt +self.fe_TP_FN

        full_prec = float(full_TP / full_TP_FP)
        full_recall =float(full_TP / full_TP_FN)
        full_f1 = 2 * full_prec * full_recall / (full_prec + full_recall+0.000001)

        print(" frame acc: %.6f " %  frame_acc)
        print(" fe_prec: %.6f " % fe_prec)
        print(" fe_recall: %.6f " % fe_recall)
        print(" fe_f1: %.6f " % fe_f1)
        print('================full struction=============')
        print(" full_prec: %.6f " % full_prec)
        print(" full_recall: %.6f " % full_recall)
        print(" full_f1: %.6f " % full_f1)

        return (frame_acc,fe_prec,fe_recall,fe_f1,full_prec,full_recall,full_f1)




