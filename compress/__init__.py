from .dummy_modules import NoCompression

def get_compressor(*args, compress_method='none', **kargs):
    if compress_method == 'none':
        return NoCompression(*args, **kargs)
    else:
        raise Exception('unknown compression method')