"""Iterate over submodules, adding args from each."""


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()

    from os.path import dirname, basename, isfile
    import glob
    import importlib
    module_names = glob.glob(dirname(__file__) + "/*.py")
    module_names = [
        basename(f)[:-3] for f in module_names
        if isfile(f) and not f.endswith('__init__.py')
    ]

    for n in module_names:
        m = importlib.import_module('.' + n, __name__)
        assert hasattr(m, 'add_args')
        m.add_args(parser)
    return parser


parser = make_parser()
parser.add_argument(
    '-f',
    '--file',
    type=str,
    default='',
    help='dummy for jupyter')
args = parser.parse_args()

from fenics import *

if args.verbose:
    set_log_level(20)
else:
    set_log_level(30)
