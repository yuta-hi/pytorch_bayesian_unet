"""
Read/Write Meta Image
"""

import re
import numpy as np
import zlib
import os
import copy
import warnings

_metatype2dtype_table = {
    'MET_CHAR':'i1',
    'MET_UCHAR':'u1',
    'MET_SHORT':'i2',
    'MET_USHORT':'u2',
    'MET_INT':'i4',
    'MET_UINT':'u4',
    'MET_LONG':'i8',
    'MET_ULONG':'u8',
    'MET_FLOAT':'f4',
    'MET_DOUBLE':'f8'
    }
_dtype2metatype_table = {
    'int8':'MET_CHAR',
    'uint8':'MET_UCHAR',
    'int16':'MET_SHORT',
    'uint16':'MET_USHORT',
    'int32':'MET_INT',
    'uint32':'MET_UINT',
    'int64':'MET_LONG',
    'uint64':'MET_ULONG',
    'float32':'MET_FLOAT',
    'float64':'MET_DOUBLE'
    }

def _str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    raise ValueError('Non boolean string')

def _str2array(string):
    for t in [int,float,_str2bool]:
        try:
            l = [t(e) for e in string.split()]
            if len(l) > 1:
                return l
            else:
                return l[0]
        except:
            continue
    return string

def _array2str(array):
    if isinstance(array, str):
        return array
    elif isinstance(array, int):
        return str(array)
    else:
        return ' '.join([str(e) for e in array])


def read_header(filename):
    """Read meta image header.

    :param str filename: Image filename with extension mhd or mha.
    :return: meta data dictionary.
    :rtype: dict
    """
    header = {}
    with open(filename, 'rb') as f:
        meta_regex = re.compile('(.+) = (.*)')
        for line in f:
            line = line.decode('ascii')
            if line == '\n': # empty line
                continue
            match = meta_regex.match(line)
            if match:
                header[match.group(1)] = match.group(2).rstrip()
                if match.group(1) == 'ElementDataFile':
                    break
            else:
                raise RuntimeError('Bad meta header line : ' + line)
    header = {key:_str2array(value) for (key, value) in header.items()} #convert string into array if possible
    return header

def _get_dim(header):
    dim = header['DimSize']
    if ('ElementNumberOfChannels' in header):
        dim = [header['ElementNumberOfChannels']] + dim
    if not hasattr(dim, '__len__'):
        dim = [dim]
    return dim

def read_memmap(filename):
    """Read Meta Image as a memory-map.

    :param str filename: Image filename with extension mhd or mha.
    :return: ND image and meta data.
    :rtype: (numpy.memmap, dict)
    :raises: RuntimeError if image data is compressed
    """
    header = read_header(filename)
    data_is_compressed = 'CompressedData' in header and header['CompressedData']
    if data_is_compressed:
        raise RuntimeError('Memory-map cannot be created for compressed data.')
    dtype = np.dtype(_metatype2dtype_table[header['ElementType']])
    data_filename = header['ElementDataFile']
    if data_filename == 'LOCAL': #mha
        numel = np.prod(np.array(_get_dim(header)))
        data_size = numel * dtype.itemsize
        offset = os.path.getsize(filename) - data_size
        data_filename = filename
    else:
        offset = 0
        if not os.path.isabs(data_filename): # data_filename is relative
            data_filename = os.path.join(os.path.dirname(filename), data_filename)
    dim = _get_dim(header)
    return np.memmap(data_filename, dtype=dtype, mode='r', shape=tuple(dim[::-1]), offset=offset), header

def read(filename):
    """Read Meta Image.

    :param str filename: Image filename with extension mhd or mha.
    :return: ND image and meta data.
    :rtype: (numpy.ndarray, dict)
    """
    header = read_header(filename)
    data_is_compressed = 'CompressedData' in header and header['CompressedData']
    data_filename = header['ElementDataFile']
    if data_filename == 'LOCAL': #mha
        if data_is_compressed:
            data_size = header['CompressedDataSize']
        else:
            numel = np.prod(np.array(_get_dim(header)))
            data_size = numel * np.dtype(_metatype2dtype_table[header['ElementType']]).itemsize
        seek_size = os.path.getsize(filename) - data_size
        with open(filename, 'rb') as f:
            f.seek(seek_size)
            data = f.read()
    else: #mhd
        if not os.path.isabs(data_filename): # data_filename is relative
            data_filename = os.path.join(os.path.dirname(filename), data_filename)
        with open(data_filename,  'rb') as fimage:
            data = fimage.read()
    if data_is_compressed:
        try:
            import pylibdeflate
            numel = np.prod(np.array(_get_dim(header)))
            decompressed_size = numel * np.dtype(_metatype2dtype_table[header['ElementType']]).itemsize
            data = pylibdeflate.zlib_decompress(data, decompressed_size)
        except:
            data = zlib.decompressobj().decompress(data)
    data = np.frombuffer(data,dtype=np.dtype(_metatype2dtype_table[header['ElementType']]))
    dim = _get_dim(header)
    image = np.reshape(data,list(reversed(dim)),order='C')
    ret = image.copy()
    ret.setflags(write=1)
    del image
    return ret, header

_default_header = {
    'ObjectType':'Image',
    'BinaryData':'True',
    'BinaryDataByteOrderMSB':'False'
        }

_no_compression_types = set(['float32','float64']) #takes long time to compress
def _is_compression_preferable(np_dtype):
    return not (np_dtype in _no_compression_types)

def _check_header_sanity(header):
    spacing = _str2array(header['ElementSpacing'])
    if hasattr(spacing, '__len__'):
        n_dims_spacing = len(spacing)
    else:
        n_dims_spacing = 1
    if n_dims_spacing != int(header['NDims']):
        warnings.warn('The number of elements of "ElementSpacing" doesn\'t match "NDims". {0} vs {1}'.format(n_dims_spacing,header['NDims']), stacklevel=3)

def write(filename, image, header={}):
    """Write Meta Image.

    :param str filename: Image filename with extension mhd or mha.
    :param numpy.ndarray image: Image to be written.
    :param dict [header]: (optional) Meta data for the image.
    """
    if image.dtype == np.bool:
        image = image.astype(np.uint8)
    # Construct header
    h = copy.deepcopy(_default_header)
    h['ElementSpacing'] = np.ones(image.ndim-1) if ('ElementNumberOfChannels') in header else np.ones(image.ndim) # default spacing
    h['CompressedData'] = _is_compression_preferable(image.dtype.name) #default compression option
    # Merge default and given headers
    h.update(header)
    # Set image dependent meta data
    h['NDims'] = len(image.shape)
    h['ElementType'] = _dtype2metatype_table[image.dtype.name]
    if ('ElementNumberOfChannels') in h:
        h['ElementNumberOfChannels'] = image.shape[-1]
        h['DimSize'] = reversed(image.shape[:-1])
        h['NDims'] -= 1
    else:
        h['DimSize'] = reversed(image.shape)
    h.pop('ElementDataFile',None) #delete 'ElementDataFile'
    h.pop('CompressedDataSize', None)
    h = {key:_array2str(value) for (key, value) in h.items()} #convert array into string if possible
    filename_base, file_extension = os.path.splitext(filename)
    compress_data = (h['CompressedData'] == 'True') #boolean variable for convenience
    if (file_extension == '.mhd'):
        if (compress_data):
            data_filename = filename_base + '.zraw'
        else:
            data_filename = filename_base + '.raw'
    else:
        if (file_extension != '.mha'):
            warnings.warn('Unknown file extension "{0}". Saving as a .mha file.'.format(file_extension), stacklevel=2)
        data_filename = 'LOCAL'
    data = np.ascontiguousarray(image)
    if (compress_data):
        try:
            import pylibdeflate
            data = pylibdeflate.zlib_compress(data)
        except:
            data = zlib.compress(data)
        h['CompressedDataSize'] = str(len(data))
    _check_header_sanity(h)
    with open(filename, 'w') as f:
        #write first two meta data
        f.write('ObjectType = '+h.pop('ObjectType')+'\n')
        f.write('NDims = '+h.pop('NDims')+'\n')
        for key, value in h.items(): #write other meta data
            f.write(key+' = '+value+'\n')
        f.write('ElementDataFile = '+os.path.basename(data_filename)+'\n') #write last meta data
        if data_filename == 'LOCAL':
            # reopen file in binary mode
            f.close()
            with open(filename, 'ab') as fdata:
                fdata.write(data)
        else:
            with open(data_filename, 'wb') as fdata:
                fdata.write(data)
