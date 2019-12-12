"""Args to define surrogate model hyperparameters"""


def add_args(parser):
    parser.add_argument(
        '--surrogate_type',
        help='composed, fenics, neural (or others if we add them)',
        default='neural',
        type=str)
    parser.add_argument(
        '--boundary_representation',
        help='spline or fem',
        default='spline',
        type=str
    )
    parser.add_argument(
        '--net_type', help='ring or ffn', default='ffn', type=str)
    parser.add_argument(
        '--use_bias', help='use biases in nets', default=False,
        action='store_true')
    parser.add_argument(
        '--solve_optimizer', help='adam or sgd or lbfgs', default='lbfgs', type=str)
    parser.add_argument(
        '--solve_steps', help='steps for adam or sgd', default=1000, type=int)
    parser.add_argument(
        '--solve_lbfgs_steps', help='steps for lbfgs', default=20, type=int)
    parser.add_argument(
        '--fenics_surrogate_dim',
        help='Mesh side length of Fenics surrogate for composing with neural',
        default=10,
        type=int)
    parser.add_argument(
        '--ffn_layer_sizes',
        help='Layer sizes for feed forward net',
        default='[512, 1024, 512, 256]',
        type=str)
    parser.add_argument(
        '--ringnet_layer_sizes',
        help='Layer sizes for ring net (in channels)',
        default='[32, 32, 32, 32, 32]',
        type=str)
