import argparse

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', '1', 'yes', 'y'}:
        return True
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
