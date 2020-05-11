"""metamaterial.py arguments"""


def add_args(parser):
    parser.add_argument(
        '--L0', help='length between pore centers', type=float, default=0.5)
    parser.add_argument(
        '--porosity',
        help='% material removed for pore, in [0, 1]',
        type=float,
        default=0.5)
    parser.add_argument(
        '--c1', help='low-freq param for pore shape', type=float, default=0.0)
    parser.add_argument(
        '--c2', help='high-freq param for pore shape', type=float, default=0.0)
    parser.add_argument(
        '--metamaterial_mesh_size',
        help='finite elements along one edge of cell; '
        ' Overvelde&Bertoldi use about sqrt(1000)',
        type=int,
        default=80)
    parser.add_argument(
        '--pore_radial_resolution',
        help='num points used to define geometry of pore boundary',
        type=int,
        default=120)
    parser.add_argument(
        '--n_cells',
        help='number cells on one side of ref volume',
        type=int,
        default=2)
    parser.add_argument(
        '--young_modulus',
        help='young\'s modulus of base material',
        type=float,
        default=100)

    # When changing 0.3 to 0.49, convergence is harder
    parser.add_argument(
        '--poisson_ratio',
        help='poisson\'s ratio of base material',
        type=float,
        default=0.3)

    parser.add_argument(
        '--padding',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--gradient',
        action='store_true',
        default=False
    )