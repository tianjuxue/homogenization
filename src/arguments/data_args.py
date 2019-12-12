"""Define how to collect data"""


def add_args(parser):
    parser.add_argument(
        '--sample_c',
        help='sample c1, c2. else take mean',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--c1_low',
        help='minimum low-freq param for pore shape',
        type=float,
        default=-0.1)
    parser.add_argument(
        '--c1_high',
        help='maximum low-freq param for pore shape',
        type=float,
        default=0.1)
    parser.add_argument(
        '--c2_low',
        help='minimum high-freq param for pore shape',
        type=float,
        default=-0.1)
    parser.add_argument(
        '--c2_high',
        help='maximum high-freq param for pore shape',
        type=float,
        default=0.1)
    parser.add_argument(
        '--min_feature_size',
        help='minimum distance between pore boundaries = minimum '
        'width of a ligament / material section in structure. We '
        'also use this as minimum width of pore.',
        type=float,
        default=0.15)
    parser.add_argument(
        '--boundary_freq_scale',
        type=float,
        help='maximum frequency scale for boundary random fourier fn',
        default=1.0)
    parser.add_argument(
        '--boundary_amp_scale',
        type=float,
        help='maximum amplitude scale for boundary random fourier fn,',
        default=1.0)
    parser.add_argument(
        '--spline_scale',
        type=float,
        help='maximum amplitude scale for spline (stddev of Gaussian)',
        default=0.2)
    parser.add_argument(
        '--spline_force_scale',
        type=float,
        help='maximum amplitude scale for spline for force (stddev of Gaussian)',
        default=0.01)
    parser.add_argument(
        '--anneal_steps',
        type=int,
        help='number of anneal steps for data gathering',
        default=100)
    parser.add_argument(
        '--metamaterial_save_path',
        type=str,
        help='path to save',
        default='saved_data_mm')
    parser.add_argument(
        '--poisson_save_path',
        type=str,
        help='path to save',
        default='saved_data_poisson')
    parser.add_argument(
        '--metamaterial_spline_save_path',
        type=str,
        help='path to save',
        default='saved_spline_data_mm')
    parser.add_argument(
        '--poisson_spline_save_path',
        type=str,
        help='path to save',
        default='saved_spline_data_poisson')
    parser.add_argument(
        '--metamaterial_spline_cdeploy_path',
        type=str,
        help='where to save deployment examples',
        default=None
    )
    parser.add_argument(
        '--metamaterial_bV_dim',
        type=int,
        help='side length of bV unitsquaremesh',
        default=5)
    parser.add_argument(
        '--poisson_bV_dim',
        type=int,
        help='side length of bV unitsquaremesh',
        default=5)
    parser.add_argument(
        '--query_V_dim',
        type=int,
        help='side length of qV',
        default=200)
    parser.add_argument(
        '--data_V_dim',
        type=int,
        help='side length of dV unitsquaremesh',
        default=100)
