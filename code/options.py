import sys
import argparse
import pprint

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--train',
            type=str,
            default='')
    argparser.add_argument('--dev',
            type=str,
            default='')
    argparser.add_argument('--test',
            type=str,
            default='')
    argparser.add_argument('--online_testing',
            type=bool,
            default=False)
    argparser.add_argument('--output',
            type=str,
            default='')
    argparser.add_argument('--vocab',
            type=str,
            default='')
    argparser.add_argument('--embedding',
            type=str,
            default='')
    argparser.add_argument('--model',
            type=str,
            default='')
    argparser.add_argument('--load_model',
            type=bool,
            default=False)

    argparser.add_argument('--batch_size',
            type=int,
            default=64)
    argparser.add_argument('--max_epochs',
            type=int,
            default=20)
    argparser.add_argument('--steps_per_checkpoint',
            type=int,
            default=1000)
    argparser.add_argument('--max_seq_length',
            type=int,
            default=20)
    argparser.add_argument('--max_train_size',
            type=int,
            default=-1)

    argparser.add_argument('--beam',
            type=int,
            default=1)
    argparser.add_argument('--dropout_keep_prob',
            type=float,
            default=0.5)
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_y',
            type=int,
            default=200)
    argparser.add_argument('--dim_z',
            type=int,
            default=500)
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.0005)
    #argparser.add_argument('--learning_rate_decay',
    #        type=float,
    #        default=0.5)
    argparser.add_argument('--rho',                 # loss_rec + rho * loss_adv
            type=float,
            default=1)
    argparser.add_argument('--gamma_init',          # softmax(logit / gamma)
            type=float,
            default=0.1)
    argparser.add_argument('--gamma_decay',
            type=float,
            default=1)
    argparser.add_argument('--gamma_min',
            type=float,
            default=0.1)
    argparser.add_argument('--filter_sizes',
            type=str,
            default='1,2,3,4,5')
    argparser.add_argument('--n_filters',
            type=int,
            default=128)

    args = argparser.parse_args()

    print('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('------------------------------------------------')

    return args
