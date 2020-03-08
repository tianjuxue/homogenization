""""Design problem related"""


def add_args(parser):
    parser.add_argument(
        '--checkpoints_path_shear',
        type=str,
        help='path to save',
        default='saved_checkpoints_shear')

    parser.add_argument(
        '--checkpoints_path_normal',
        type=str,
        help='path to save',
        default='saved_checkpoints_normal')

    parser.add_argument(
        '--checkpoints_path',
        type=str,
        help='path to save',
        default='saved_checkpoints')