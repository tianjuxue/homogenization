"""Args to define training and optimizer hyperparameters"""


def add_args(parser):
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
