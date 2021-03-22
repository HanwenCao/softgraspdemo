def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_k':
        from .ggcnn2_k import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_k_plus':
        from .ggcnn2_k_plus import GGCNN2
        return GGCNN2
    elif network_name == 'ggcnn2_k_plus_pro':
        from .ggcnn2_k_plus_pro import GGCNN2
        return GGCNN2
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
