import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')


    parser.add_argument('--data', type=str, default='data/train/',
                        help='Dataset location')
    parser.add_argument('--model', type=str, default='MultiVAE',
                        help='Model name')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer name')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=4948,
                        help='random seed number')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model/model.pt',
                        help='path to save the final model')
    parser.add_argument('--mode', choices=['tuning', 'submission'], default='tuning',
                        help='Train dataset range (tuning: split / submission: full)')
    
    args = parser.parse_args()

    return args