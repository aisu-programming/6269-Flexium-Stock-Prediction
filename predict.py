''' Libraries '''
import argparse
import os
import torch

from exp.exp_informer import Exp_Informer


''' Parameters '''
MODEL_DIRECTORY = "results/2021-03-16_03.10.58_bs50_sl200_ll100_pl23_sTrue_do0.05_fc5_dtTrue_test_1"


''' Functions '''
def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_len', type=int, default=400)
    parser.add_argument('--label_len', type=int, default=300)
    parser.add_argument('--pred_len', type=int, default=23)
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
    parser.add_argument('--scale', type=bool, default=False, help='scale the dataset (Add by Aisu)')  # Add by Aisu
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--lradj', type=str, default='type1')

    parser.add_argument('--distil', action='store_false')
    parser.add_argument('--output_attention', action='store_true')
    args = parser.parse_args()
    args.model = 'informer'
    args.data = 'custom'
    args.root_path = './data/custom/'
    args.data_path = '6269_rm_front.csv'
    args.features = 'MS'
    args.target = 'Close'
    args.freq = 'd'
    args.enc_in = 4
    args.dec_in = 4
    args.c_out = 1
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 2048
    args.factor = 5
    args.attn = 'prob'
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 300
    args.patience = 20
    args.tag = 'test'
    args.loss = 'mse'
    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0

    return args


''' Execution '''
if __name__ == '__main__':

    torch.cuda.empty_cache()

    args = get_args()
    print('Args in experiment:')
    print(args)

    exp = Exp_Informer(args)
    exp.model.load_state_dict(torch.load(f"{MODEL_DIRECTORY}/checkpoint.pth"))
    exp.model.eval()
    
    exp.predict(MODEL_DIRECTORY)