""""Design problem related"""


def add_args(parser):

    parser.add_argument(
        '--design_save_path',
        type=str,
        help='path to save',
        default='saved_data_design')
    parser.add_argument(
        '--design_save_path_medium',
        type=str,
        help='path to save medium size of the data',
        default='saved_data_design_medium_size')    
    parser.add_argument(
        '--design_save_path_small',
        type=str,
        help='path to save small size of the data',
        default='saved_data_design_small_size') 
    parser.add_argument(
        '--design_checkpoints_path',
        type=str,
        help='path to save',
        default='saved_checkpoints_design')


    parser.add_argument(
        '--data_path_shear',
        type=str,
        help='path to save',
        default='saved_data_shear')

    parser.add_argument(
        '--data_path_normal',
        type=str,
        help='path to save',
        default='saved_data_normal')

    parser.add_argument(
        '--data_path_dummy',
        type=str,
        help='path to save',
        default='saved_data_dummy')

    parser.add_argument(
        '--data_path_dummy_modified',
        type=str,
        help='path to save',
        default='saved_data_dummy_modified')

    parser.add_argument(
        '--data_path_integrated_regular',
        type=str,
        help='path to save',
        default='saved_data_integrated_regular')

    parser.add_argument(
        '--data_path_integrated_random',
        type=str,
        help='path to save',
        default='saved_data_integrated_random')

    parser.add_argument(
        '--data_path_corner_regular',
        type=str,
        help='path to save',
        default='saved_data_corner_regular')    


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
        '--checkpoints_path_dummy',
        type=str,
        help='path to save',
        default='saved_checkpoints_dummy')