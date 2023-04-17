import sys
from main import main
import argparse
from config import load_config
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## Parameter setting
    current_dir = sys.path[0]
    parser.add_argument("--path", type=str, default=current_dir)
    parser.add_argument("--data_path", type=str, default="/data/", help="Path of datasets.")
    parser.add_argument("--dataset", type=str, default="Cora", help="Name of datasets.")

    parser.add_argument("--device", type=str, default="2", help="Device: cuda:num or cpu")
    parser.add_argument("--use_seed", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=40, help="Random seed, default is 42.")
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument("--norm", action='store_true', default=True, help="Normalize the feature.")

    args =parser.parse_args()

    dataset = {1:"ALOI",2:"MNIST10k",3:"NUSWIDE",4:"scene15"}

    select_dataset = [1]
    for i in select_dataset:
        config = load_config('./config/' + dataset[i])
        args.gamma = config['gamma']
        args.block = config['block']
        args.epoch = config['epoch']
        args.thre = config['thre']
        args.lr = config['lr']
        main(dataset[i], args)

