from data.seq7_dataset import Seq7TrainDataset, Seq7ValDataset, Seq7TestDataset
from data.baist_dataset import BaistTrainDataset, BaistValDataset, BaistTestDataset


def get_datasets(**kwargs):
    train_set = Seq7TrainDataset(
        metadata_root=kwargs['metadata_root'],
        target_frames=kwargs['target_frames'],
        limit=kwargs['train_limit'],
    )

    val_set = Seq7ValDataset(
        metadata_root=kwargs['metadata_root'],
        target_frames=kwargs['target_frames'],
        limit=kwargs['val_limit'],
    )

    tral_set = Seq7TrainDataset(
        metadata_root=kwargs['metadata_root'],
        target_frames=kwargs['target_frames'],
        limit=kwargs['val_limit'],
    )

    test_set = Seq7TestDataset(
        metadata_root=kwargs['metadata_root'],
        target_frames=kwargs['target_frames'],
        limit=kwargs['test_limit'],
    )

    return train_set, val_set, tral_set, test_set

def get_datasets_b(**kwargs):
    train_set = BaistTrainDataset(
        metadata_root=kwargs['metadata_root'],
        target_frames=kwargs['target_frames'],
        limit=kwargs['train_limit']
    )

    val_set = BaistValDataset(
        metadata_root=kwargs['metadata_root'],
        target_frames=kwargs['target_frames'],
        limit=kwargs['val_limit']
    )

    tral_set = BaistTrainDataset(
        metadata_root=kwargs['metadata_root'],
        target_frames=kwargs['target_frames'],
        limit=kwargs['val_limit'],
    )

    test_set = BaistTestDataset(
        metadata_root=kwargs['metadata_root'],
        target_frames=kwargs['target_frames'],
        limit=kwargs['test_limit']
    )

    return train_set, val_set, tral_set, test_set
