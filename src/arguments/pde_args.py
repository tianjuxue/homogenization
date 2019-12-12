"""pde.py arguments"""


def add_args(parser):
    parser.add_argument(
        '--relaxation_parameter',
        default=0.8,
        type=float,
        help='relaxation parameter for Newton')
    parser.add_argument(
        '--nonlinear_solver',
        default='newton',
        type=str,
        help='Nonlinear solver: newton or snes'
    )
    parser.add_argument(
        '--snes_method',
        default='qn',
        type=str,
        help='newtontr, newtonls, qn, ...'
    )
    parser.add_argument(
        '--linear_solver',
        default='petsc',
        type=str,
        help='Newton linear solver')
    parser.add_argument(
        '--preconditioner',
        default='ilu',
        type=str,
        help='Preconditioner')
    parser.add_argument(
        '--max_newton_iter',
        default=1000,
        type=int,
        help='Newton maximum iters')
    parser.add_argument(
        '--max_snes_iter',
        default=1000,
        type=int,
        help='Newton maximum iters')
    parser.add_argument(
        '--adaptive',
        default=False,
        action='store_true',
        help='Use adaptive solver')
    parser.add_argument(
        '--manual_solver',
        default=False,
        action='store_true',
        help='Use homemade Newton solver')
