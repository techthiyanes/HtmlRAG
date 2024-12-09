import os
import mmap
import fcntl
import ctypes
import struct
import pickle


__all__ = ['RecordIO', 'Record']


class RecordIO(object):
    magic = b'recordio'

    def __init__(self, path, flag='r', shm_dir=None):
        assert flag in ['r', 'w'], 'Invalid flag {}'.format(flag)
        self.path, self.flag = path, flag
        self.shm_dir = shm_dir
        if shm_dir in (True, 'True', 'true'):
            self.shm_dir = '/dev/shm/recordio'
        else:
            self.shm_dir = shm_dir

        if self.flag == 'r':
            if self.shm_dir:
                shm_path = os.path.join(self.shm_dir, self.path.lstrip('/'))
                os.makedirs(os.path.dirname(shm_path), exist_ok=True)
                with open(shm_path + '.lock', 'w') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)  # 加锁
                    if not os.path.exists(shm_path):
                        print(f'copying {self.path} -> {shm_path}', flush=True)
                        os.system(f'cp {self.path} {shm_path}')
                    fcntl.flock(lock, fcntl.LOCK_UN)  # 释放
                self._fd = open(shm_path, 'rb+')
            else:
                self._fd = open(self.path, 'rb+')
            assert self._fd.read(len(self.magic)) == self.magic
            size = struct.unpack('I', self._fd.read(4))[0]
            self._mmap = mmap.mmap(self._fd.fileno(), 0)
            self._offset = (ctypes.c_ulong * size).from_buffer(self._mmap, self._fd.tell())
            self._base = self._fd.tell() + size * 8
        else:
            self._fd = open(self.path + '.w', 'wb')
            self._offset = []

    def __len__(self):
        return len(self._offset)

    def __getitem__(self, idx):
        if self.flag != 'r':
            raise IOError('only support getitem when reading.')
        elif isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        elif idx < 0:
            return self[idx + len(self)]
        else:
            idx = int(idx)
            return self._mmap[(self._offset[idx - 1] if idx else self._base):self._offset[idx]]

    def append(self, buffer):
        if self.flag != 'w':
            raise IOError('only support append when writing.')
        elif isinstance(buffer, bytes):
            self._fd.write(buffer)
            self._offset.append(self._fd.tell())
        else:
            raise IOError('only support append bytes')

    def __str__(self):
        return self.__class__.__name__ + '({})'.format(', '.join([
            'path={}'.format(self.path),
            'size={}'.format(len(self)),
            'flag={}'.format(self.flag),
            'shm_dir={}'.format(self.shm_dir)
        ]))

    def close(self):
        if hasattr(self, '_fd'):
            self._fd.close()
            del self._fd
            if self.flag == 'w':
                with open(self.path, 'wb') as f:
                    f.write(self.magic)
                    f.write(struct.pack('I', len(self._offset)))
                    base = f.tell() + len(self._offset) * 8
                    for offset in self._offset:
                        f.write(struct.pack('Q', base + offset))
                    assert base == f.tell()
                    f.write(open(self.path + '.w', 'rb').read())
                os.remove(self.path + '.w')

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Record(object):
    '''A readonly pickle-able wrapper for RecordIO'''
    def __init__(self, path: str, shm_dir=None, auto_pickle=False):
        self.path = os.path.expanduser(path)
        self.shm_dir = shm_dir
        self.auto_pickle = auto_pickle
        self.io = RecordIO(self.path, shm_dir=self.shm_dir)

    def __getitem__(self, idx):
        buffer = self.io[idx]
        return pickle.loads(buffer) if self.auto_pickle else buffer

    def __len__(self):
        return len(self.io)

    def __str__(self):
        return "{}(size={}, path='{}')".format(self.__class__.__name__, len(self), self.path)

    def __getstate__(self):
        return {
            'path': self.path,
            'shm_dir': self.shm_dir,
            'auto_pickle': self.auto_pickle
        }

    def __setstate__(self, state):
        self.path = state['path']
        self.shm_dir = state['shm_dir']
        self.auto_pickle = state['auto_pickle']
        self.io = RecordIO(self.path, shm_dir=self.shm_dir)

