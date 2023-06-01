from testing.generate_order import OrderGenerator
from utils import training_utils


def parse_args():
    parser = training_utils.get_basic_parser()

    # Additional parameters
    parser.add_argument('--model_name', default='HyperCUT', type=str, required=True, help='Model name')
    parser.add_argument('--f_func', type=str, default='ResnetIm2Vec', required=True, help='function f')
    parser.add_argument('--g_func', type=str, default='Concat', required=True, help='function g')
    parser.add_argument('--out_dim', type=int, default=128, help='Motion vector dimension')
    parser.add_argument('--save_path', type=str, required=True, help='save path')

    args = parser.parse_args()

    num_frames = 2
    args.use_flow = (args.g_func == 'flow')
    args.target_frames = [1, 2, 3, 4, 5, 6, 7]
    args.model_kwargs = {
        'f_func': args.f_func,
        'g_func': args.g_func,
        'num_frames': num_frames,
        'out_dim': args.out_dim,
    }

    return args


if __name__ == '__main__':
    args = parse_args()

    tester = OrderGenerator(args)
    tester.test()
