import argparse
import torch


def get_args():
    
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--algo', default='pvf', help='algo (default: pvf, choices: pvf, mrpvf)')
    parser.add_argument('--num_eigvals', type=int, default=80, help='number of eigen vectors to use (default: 80)')
    parser.add_argument('--decomp_type', default='eig', help='decomposition (default: eig, choices: eig, svd)')
    parser.add_argument('--transfer', type=bool, default=False, help='whether to run the transfer setting or not (default: False)')
    parser.add_argument('--stochastic', type=bool, default=False, help='whether actions are stochastic or not (default: False)')
    parser.add_argument('--run_index', type=int, default=0, help='index of the run for saving weights and logs for multiple simultaneous transfer experiments (default: 0)')

    args = parser.parse_args()

    return args
