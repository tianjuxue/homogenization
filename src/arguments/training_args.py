"""Args to define training and optimizer hyperparameters"""


def add_args(parser):
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--wd', help='Weight decay', type=float, default=0.0)
    parser.add_argument(
        '--J_weight', help='Weight on Jacobian loss', type=float, default=0.0)
    parser.add_argument(
        '--H_weight', help='Weight on Hessian loss', type=float, default=0.0)
    parser.add_argument(
        '--deploy_loss_weight', help='Weight on deploy loss', type=float, default=0.0)
    parser.add_argument(
        '--val_fraction', help='Validation fraction', type=float, default=0.1)
    parser.add_argument(
        '--clip_grad_norm', help='Norm for gradient clipping', type=float, default=None)
    parser.add_argument(
        '--max_train_steps',
        help='Maximum training steps',
        type=int,
        default=int(1e7))
    parser.add_argument(
        '--results_dir',
        help='Dir for tensorboard and other output',
        type=str,
        default='results')
    parser.add_argument(
        '--experiment_name',
        help='Name of experiment run',
        type=str,
        default='default')
    parser.add_argument(
        '--batch_size', help='Batch size', type=int, default=100)
    parser.add_argument(
        '--deploy_n', help='N examples to deploy on', type=int, default=1)
    parser.add_argument(
        '--deploy_every', help='N steps to deploy', type=int, default=10000)
    parser.add_argument(
        '--max_files', help='Max data files to train on',
        type=int, default=100000)
