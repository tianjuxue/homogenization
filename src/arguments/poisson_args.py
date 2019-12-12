"""poisson.py arguments"""


def add_args(parser):
    parser.add_argument(
        '--poisson_mesh_size',
        help='Number of elements on one side of square mesh',
        type=int,
        default=25)
