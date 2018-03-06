import torch


def mkTensor(field, capacity, cuda=False)
    assert(isinstance(field, Field))
    # Not case sensitive
    typename = typename.lower()
    doubleNames = ['double', 'float64']
    floatNames = ['float', 'float32']
    halfNames = ['half', 'float16']
    byteNames = ['byte', 'uint8', 'bool']
    charNames = ['char', 'int8']
    shortNames = ['short', 'int16']
    intNames = ['int', 'int32']
    longNames = ['long', 'int64']

    fullDims = [capacity] + field.dims
    if field.filename in doubleNames:
        if cuda:
            return torch.cuda.DoubleTensor(*fullDims)._zero()
        return torch.DoubleTensor(*fullDims)._zero()

    if field.filename in floatNames:
        if cuda:
            return torch.cuda.FloatTensor(*fullDims)._zero()
        return torch.FloatTensor(*fullDims)._zero()

    if field.filename in halfNames:
        if cuda:
            return torch.cuda.HalfTensor(*fullDims)._zero()
        return torch.HalfTensor(*fullDims)._zero()

    if field.filename in byteNames:
        if cuda:
            return torch.cuda.ByteTensor(*fullDims)._zero()
        return torch.ByteTensor(*fullDims)._zero()

    if field.filename in charNames:
        if cuda:
            return torch.cuda.CharTensor(*fullDims)._zero()
        return torch.CharTensor(*fullDims)._zero()

    if field.filename in shortNames:
        if cuda:
            return torch.cuda.ShortTensor(*fullDims)._zero()
        return torch.ShortTensor(*fullDims)._zero()

    if field.filename in byteNames:
        if cuda:
            return torch.cuda.IntTensor(*fullDims)._zero()
        return torch.IntTensor(*fullDims)._zero()

    if field.filename in byteNames:
        if cuda:
            return torch.cuda.LongTensor(*fullDims)._zero()
        return torch.LongTensor(*fullDims)._zero()

class Field(object):
    def __init__(typename, dims, name):
        typename = typename.lower()
        Names = ['double', 'float64', 
                 'float',  'float32', 
                 'half',   'float16',
                 'byte',   'uint8', 'bool', 
                 'char',   'int8',
                 'short',  'int16',
                 'int',    'int32', 
                 'long',   'int64']
        if not typename in Names:
            raise LookupError
        field.typename = typename
        field.dims = dims
        field.name = name

class ReplayDataset(Dataset):
    """
    Dataset to for experience replay.
    Purpose: easy experience replay.
    """

    def __init__(self, fields, capacity):
        super(ReplayDataset, self).__init__()
        self.len = 0
        self.full = False
        self.capacity = capacity
        self.fields = [mkTensor(field, capacity) for field in fields]
        self.names = [field.name for field in fields]

    def __len__(self):
        return self.len

    def __add__(self, x):
        if not self.full:
            for i in range(len(self.fields)):
                self.fields[i][self.len] = x[i]
            self.len += 1
            self.full = self.len==self.capacity
        else:
            j = random.randint(0, self.len)
            for i in range(len(self.fields)):
                self.fields[i][j] = x[i]

    def __getitem__(self, idx):
        l = [field[idx] for field in self.fields]
        return *l

