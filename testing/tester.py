import torch
from torch.utils.data import DataLoader

from models import get_model
from data import get_datasets


class Tester:
    def __init__(self, args):
        self.args = args

        if args.exp_name is None:
            self.exp_name = args.dataset_name + '_' + args.model_name
        else:
            self.exp_name = args.exp_name
        self.device = torch.device("cuda")

        self.load_dataloader()

        self.model = get_model(args.model_name, args.model_kwargs).to(self.device)
        self.model.load_state_dict(torch.load(self.args.pretrained_path))

        # self.save_path = args.save_path

    def load_dataloader(self):
        _, _, _, test_set = get_datasets(
            **vars(self.args)
        )

        print(f'Number of test data points: {len(test_set)}')

        self.dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=self.args.num_workers)

    def test(self):
        self.model.test(self.dataloader)
