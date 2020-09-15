
import argparse
import os

data_dir ='./'


class PLMConfig:
    MODEL_PATH = 'uncased_L-12_H-768_A-12'
    VOCAB_PATH = f'{MODEL_PATH}/vocab.txt'
    CONFIG_PATH = f'{MODEL_PATH}/bert_config.json'

def get_opt():
    parser = argparse.ArgumentParser()

    # 数据集位置
    parser.add_argument('--data_path', type=str, default='parsed-v1.5/')

    # 保存模型和加载模型相关
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default='train_model')
    parser.add_argument('--pretrain_model', type=str, default='')
    parser.add_argument('--save_model_path', type=str, default='./models_ft/model_frame_first.bin')

    # 训练相关
    parser.add_argument('--lr', type=float, default='3e-5')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_model_freq', type=int, default=1)  # 保存模型间隔，以epoch为单位
    parser.add_argument('--cuda', type=str, default="cuda:2:3")
    parser.add_argument('--mode', type=str, default="train")

    # 模型的一些settings
    parser.add_argument('--maxlen',type=int,default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cell_name', type=str, default='gru')
    parser.add_argument('--rnn_hidden_size',type=int,default=256)
    parser.add_argument('--decoder_emb_size',type=int,default=768)
    parser.add_argument('--decoder_hidden_size',type=int,default=768)
    parser.add_argument('--target_maxlen',type=int,default=5)
    parser.add_argument('--decoderlen',type=int,default=4)
    parser.add_argument('--frame_number',type=int,default=1019)
    parser.add_argument('--role_number',type=int,default=9634)
    parser.add_argument('--fe_padding_num',type=int,default=5)


    return parser.parse_args()







if __name__ == '__main__':
    bertconfig = BertConfig()
    print(bertconfig.CONFIG_PATH)
