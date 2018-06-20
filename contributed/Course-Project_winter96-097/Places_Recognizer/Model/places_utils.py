from keras import backend as K

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 104.006
        x[:, 1, :, :] -= 116.669
        x[:, 2, :, :] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 104.006
        x[:, :, :, 1] -= 116.669
        x[:, :, :, 2] -= 122.679
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x
