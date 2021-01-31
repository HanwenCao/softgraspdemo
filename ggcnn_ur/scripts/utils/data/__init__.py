def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'cornell_k':
        from .cornell_data_k import CornellDatasetk
        return CornellDatasetk
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'my':
        from .my_data import MyDataset
        return MyDataset
    elif dataset_name == 'my2':
        from .my_data2 import MyDataset2
        return MyDataset2
    elif dataset_name == 'my_crop':
        from .my_data_crop import MyDataset3
        return MyDataset3
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))
