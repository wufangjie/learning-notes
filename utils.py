import sys
import numpy as np
from functools import wraps


try:
    import mysql.connector
except ImportError:
    print('Warning: import mysql.connector failed', file=sys.stderr)


__all__ = ['ConnectMysqlGetCursor', 'memo',
           'conn_paipian', 'font_exec',
           'get_rank', 'get_index',
           'sort_rows', 'sort_rows_rec',
           'intersect_loop', 'intersect',
           'unique_func', 'unique_lazy']


############################## classes ##############################
class ConnectMysqlGetCursor:
    def __init__(self, database='*',
                 user='*', password='*', host='*',
                 port=3306, raise_on_warnings=True, autocommit=True, **kwargs):
        self.params = locals()
        self.params.pop('self')
        self.params.pop('kwargs')
        self.params.update(kwargs)

    def __enter__(self):
        self.conn = mysql.connector.connect(**self.params)
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, *args):
        self.cursor.close()
        self.conn.close()


############################## decorators ##############################
def memo(func): # sys.getsizeof(frozenset()) -> 224 bytes
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper._cache = cache
    return wrapper


def memo_with_kwargs(func):
    cache = {}
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    wrapper._cache = cache
    return wrapper


############################## variables ##############################
font_exec = '''
from matplotlib.font_manager import FontProperties
font_path = '/usr/share/fonts/opentype/noto/'
font_title = FontProperties(fname=font_path+'NotoSansCJK-Bold.ttc', size=32)
font_label = FontProperties(fname=font_path+'NotoSansCJK-Medium.ttc', size=24)
font_text = FontProperties(fname=font_path+'NotoSansCJK-Regular.ttc', size=20)
'''


############################## functions ##############################
def get_rank(var, method='dense'):
    """Assign ranks to data, dealing with ties appropriately.

    Ranks begin at 1.

    Parameters
    ----------
    method: {'min', 'max', 'dense', 'ordinal', 'average'}

    mostly from scipy
    """
    if method not in {'min', 'max', 'dense', 'ordinal', 'average'}:
        raise ValueError('Unknown method "{}"'.format(method))

    var = np.asarray(var)
    kind = 'mergesort' if method == 'ordinal' else 'quicksort'
    idx = np.argsort(var, kind=kind)
    n = idx.size
    rank = np.empty(n, dtype=int)
    rank[idx] = np.arange(n)

    if method == 'ordinal':
        return rank + 1

    var = var[idx]
    index = np.concatenate(([True], var[:-1] != var[1:], [True]))
    dense = np.cumsum(index)[rank]

    if method == 'dense':
        return dense

    count = np.nonzero(index)[0]

    if method == 'min':
        return count[dense - 1] + 1

    if method == 'max':
        return count[dense]

    if method == 'average':
        return .5 * (count[dense - 1] + 1 + count[dense])


def get_index(arr, mode='both'):
    """Get the start (and end) index of a unique arr[i] in a sorted array

    Parameters
    ----------
    mode: {'both' (default), other}
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        indexStart = np.nonzero(np.concatenate((
            [True], arr[:-1] != arr[1:])))[0]
    else:
        indexStart = np.nonzero(np.concatenate((
            [True], [np.any(x) for x in arr[:-1] != arr[1:]])))[0]
    if mode == 'both':
        return indexStart, np.concatenate((indexStart[1:], [len(arr)]))
    else:
        return indexStart


def sort_rows(data, keyList, stable=False):
    """Sort data (2d-array) by multi-columns, return change index.

    Notes
    -----
    1. Stable sorting.
    2. In order to sort the first column both ascending and descending,
the keyList should be 1-indexed.
    3. If data is numeric, multiple -1 is a better choice.

    Parameters
    ----------
    data    : Array to be sorted.

    keyList : sequence of integer
        The sequence of columns to sort the data. The negative value
means descending. For instance, [2, -1] means the data should be sorted
by the second column ascending, then by the first column descending.

    """
    keyList = np.asarray(keyList)[::-1]
    if np.any(keyList == 0):
        raise Exception('keyList should be 1-indexed!')

    data = np.asarray(data)
    data = data[:, abs(keyList) - 1].T
    if np.all(keyList > 0):
        return np.lexsort(data)

    return np.lexsort([row if k > 0 else -get_rank(row, 'dense')
                       for k, row in zip(keyList, data)])


def sort_rows_rec(data, keyList, kind='quicksort', copy=True):
    """Sort data (2d-array) by multi-columns, return change index.

    (Not recommended), when data is big, toooooo slow

    Notes
    -----
    1. In order to sort the first column both ascending and descending,
the keyList should be 1-indexed.
    2. If keyList have same sign or data[:, np.abs(keyList)] is numeric,
use sort_rows instead, which having better performance.

    Parameters
    ----------
    data    : Array to be sorted.

    keyList : sequence of integer
        The sequence of columns to sort the data. The negative value
means descending. For instance, [2, -1] means the data should be sorted
by the second column ascending, then by the first column descending.

    kind    : {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm. Default is 'quicksort'.

    copy    : bool, optional
        If False, data will be changed. Default is False.
    """
    def sort_rows_(data, keyList, index, indexPart):
        if keyList.size:
            col = abs(keyList[0]) - 1

            if keyList[0] > 0:
                indexNew = np.argsort(data[indexPart, col], kind=kind)
            elif kind == 'mergesort':
                indexNew = (len(indexPart) - 1 - np.argsort(
                    data[indexPart[::-1], col], kind=kind))[::-1]
            else:
                indexNew = np.argsort(data[indexPart, col], kind=kind)[::-1]

            indexNew = indexPart[indexNew]
            index[indexPart] = index[indexNew]
            data[indexPart, :] = data[indexNew, :]

            indexStart, indexEnd = get_index(data[indexPart, col])

            for k in np.nonzero(indexEnd - indexStart != 1)[0]:
                sort_rows_(data, keyList[1:], index,
                           indexPart[indexStart[k]:indexEnd[k]])

    keyList = np.asarray(keyList)
    if np.any(keyList == 0):
        raise Exception('keyList should be 1-indexed!')

    data_ = data.copy() if copy else data
    index = np.arange(len(data))
    sort_rows_(data_, keyList, index, index.copy())

    return index


def intersect_loop(var1, var2, sorted_asc=False):
    """Return (sameValue, index1, index2).

    Notes
    -----
    1. Ensure the type of var1 and var2 is ndarray
    2. Ensure the element of var1 and var2 is unique
    All of above are designed for easy to index var1 and var2,
otherwise use numpy.intersect1d instead.
    """
    var1.shape = var2.shape = -1
    len1, len2 = len(var1), len(var2)
    same, index1, index2 = [], [], []
    i = j = 0

    if not sorted_asc:
        idx1, idx2 = np.argsort(var1), np.argsort(var2)
        var1, var2 = var1[idx1], var2[idx2]

    while i < len1 and j < len2:
        if var1[i] == var2[j]:
            same.append(var1[i]), index1.append(i), index2.append(j)
            i += 1
            j += 1
        elif var1[i] < var2[j]:
            i += 1
        else:
            j += 1

    index1, index2 = np.array(index1), np.array(index2)
    if not sorted_asc:
        index1, index2 = idx1[index1], idx2[index2]

    return same, index1, index2


def intersect(var1, var2, sorted_asc=False):
    """Return (sameValue, index1, index2).

    Notes
    -----
    1. Ensure the type of var1 and var2 is ndarray
    2. Ensure the element of var1 and var2 is unique
    All of above are designed for easy to index var1 and var2,
otherwise use numpy.intersect1d instead.
    """
    var1, var2 = np.asarray(var1), np.asarray(var2)
    var1.shape = var2.shape = -1
    same = np.intersect1d(var1, var2, assume_unique=True)

    if sorted_asc:
        return (same,
                np.nonzero(np.in1d(var1, same, assume_unique=True))[0],
                np.nonzero(np.in1d(var2, same, assume_unique=True))[0])
    else:
        idx1, idx2 = np.argsort(var1), np.argsort(var2)
        var1, var2 = var1[idx1], var2[idx2]
        return (same,
                idx1[np.in1d(var1, same, assume_unique=True)],
                idx2[np.in1d(var2, same, assume_unique=True)])


def unique_func(arr, data=None, func=None, mode='index', **kwargs):
    """Aggregate data group by arr[i].

    Return (the start index or the value of each unique type,
            result applying func on each unique type data with kwargs).
           If mode == 'index' (default) return the index,
           otherwise return the unique value instead.
    Notes
    -----
    1. Ensure arr is sorted, otherwise use unique_lazy.
    2. arr[i] present one type.
    3. If data or func is None, return the count of each unique type.

    See Also
    --------
    numpy.unique
    """
    arr = np.asarray(arr)
    indexStart, indexEnd = get_index(arr)
    ret = indexStart if mode == 'index' else arr[indexStart]

    if data is None or func is None:
        return ret, indexEnd - indexStart
    else:
        return ret, [func(data[i:j], **kwargs)
                     for i, j in zip(indexStart, indexEnd)]


def unique_lazy(arr, data=None, func=None, **kwargs):
    """A lazy wrapper of unique_func"""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        idx = np.argsort(arr)
    elif arr.ndim == 2:
        idx = np.lexsort(arr.T)
    else:
        raise NotImplementedError('arr.ndim > 2')

    data = None if data is None else data[idx]
    return unique_func(arr[idx], data, func, mode='value', **kwargs)
