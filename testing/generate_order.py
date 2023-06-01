import json

import torch
from data import get_datasets
from testing.tester import Tester
from torch.utils.data import DataLoader
from tqdm import tqdm


class OrderGenerator(Tester):
    def __init__(self, args):
        super().__init__(args)

        self.metadata = []
        self.pos, self.neg = 0, 0

    def load_dataloader(self):
        train_set, val_set, _, test_set = get_datasets(**vars(self.args))

        print(f"Number of train data points: {len(train_set)}")
        print(f"Number of test data points: {len(test_set)}")
        print(f"Number of val data points: {len(val_set)}")

        self.train_dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=self.args.num_workers)
        self.val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=self.args.num_workers)
        self.test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=self.args.num_workers)

    def test_partition(self, dataloader):
        for data in tqdm(dataloader, total=len(dataloader)):
            with torch.no_grad():
                d00, _, d10, _, d20, _ = self.model(data)
                u0 = self.model.hyperplane(d00).item()
                u1 = self.model.hyperplane(d10).item()
                u2 = self.model.hyperplane(d20).item()

            metadata = {x: y[0] for x, y in data["meta"].items()}
            if u0 < 0 and u0 * u1 > 0 and u0 * u2 > 0:
                metadata["order"] = "reverse"
                self.pos += 1
            else:
                self.neg += 1
            self.metadata.append(metadata)

    def test(self):
        self.test_partition(self.train_dataloader)
        self.test_partition(self.test_dataloader)
        self.test_partition(self.val_dataloader)

        metadata = {"name": self.args.dataset_name, "frame_per_seq": 7, "data": self.metadata}

        with open(self.args.save_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(self.pos, self.neg)
