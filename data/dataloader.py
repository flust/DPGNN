from torch.utils.data import DataLoader


def construct_dataloader(config, datasets):
    param_list = [
        [*datasets],
        [config['train_batch_size']] + [config['eval_batch_size']] * 2,
        [True, False, False],
        [config['num_workers']] * 3,
        [config['pin_memory']] * 3
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
