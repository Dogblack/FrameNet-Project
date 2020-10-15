
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
    parser.add_argument('--emb_file_path', type=str, default='./glove.6B/glove.6B.200d.txt')
    parser.add_argument('--train_instance_path', type=str, default='./train_instance_dic.npy')
    parser.add_argument('--dev_instance_path', type=str, default='./dev_instance_dic.npy')
    parser.add_argument('--test_instance_path', type=str, default='./test_instance_dic.npy')
    # 保存模型和加载模型相关
    parser.add_argument('--load_instance_dic', type=bool, default=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default='train_model')
    parser.add_argument('--pretrain_model', type=str, default='')
    parser.add_argument('--save_model_path', type=str, default='./models_ft/model_syntax.bin')

    # 训练相关
    parser.add_argument('--lr', type=float, default='0.0001')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_model_freq', type=int, default=1)  # 保存模型间隔，以epoch为单位
    parser.add_argument('--cuda', type=str, default="cuda:0")
    parser.add_argument('--mode', type=str, default="train")

    # 模型的一些settings
    parser.add_argument('--maxlen',type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cell_name', type=str, default='lstm')
    parser.add_argument('--rnn_hidden_size',type=int, default=256)
    parser.add_argument('--rnn_emb_size', type=int, default=400)
    parser.add_argument('--encoder_emb_size',type=int, default=200)
    parser.add_argument('--sent_emb_size', type=int, default=200)
    parser.add_argument('--pos_emb_size',type=int, default=64)
    parser.add_argument('--rel_emb_size',type=int,default=100)
    parser.add_argument('--token_type_emb_size',type=int, default=36)
    parser.add_argument('--decoder_emb_size',type=int, default=200)
    parser.add_argument('--decoder_hidden_size',type=int, default=256)
    parser.add_argument('--target_maxlen',type=int, default=5)
    parser.add_argument('--decoderlen',type=int,default=4)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--frame_number',type=int, default=1019)
    parser.add_argument('--role_number',type=int, default=9634)
    parser.add_argument('--fe_padding_num',type=int, default=5)
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=4)

    return parser.parse_args()







if __name__ == '__main__':
    bertconfig = BertConfig()
    print(bertconfig.CONFIG_PATH)
