"""pde.py arguments"""


def add_args(parser):
    parser.add_argument(
        '--relaxation_parameter',
        default=0.8,
        type=float,
        help='relaxation parameter for Newton')
    parser.add_argument(
        '--max_newton_iter',
        default=1000,
        type=int,
        help='Newton maximum iters')
