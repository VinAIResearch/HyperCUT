from testing.check_consistency import ConsistencyChecker
from utils import training_utils


def parse_args():
    parser = training_utils.get_basic_parser()

    # Additional parameters
    parser.add_argument("--f_func", type=str, required=True, help="function f")
    parser.add_argument("--g_func", type=str, required=True, help="function g")
    parser.add_argument("--out_dim", type=int, required=True, help="Motion vector dimension")
    parser.add_argument("--target_frames", nargs="+", type=int, default=None, help="Target frames")

    args = parser.parse_args()
    args.model_name = "HyperCUT"
    num_frames = 2

    args.use_flow = args.g_func == "flow"

    args.model_kwargs = {
        "f_func": args.f_func,
        "g_func": args.g_func,
        "num_frames": num_frames,
        "out_dim": args.out_dim,
    }

    return args


if __name__ == "__main__":
    args = parse_args()

    tester = ConsistencyChecker(args)
    tester.test()
