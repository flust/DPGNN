from torch.utils.data import DataLoader


def construct_dataloader(config, datasets):
    param_list = [
        [*datasets],
        [config['train_batch_size']] * 1 + [config['eval_batch_size']] * 4,
        [True, False, False, False, False],
        [config['num_workers']] * 5,
        [config['pin_memory']] * 5
    ]
    dataloaders = [
        DataLoader(
            dataset=ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=pm
        ) for ds, bs, shuffle, nw, pm in zip(*param_list)
    ]
    return dataloaders
