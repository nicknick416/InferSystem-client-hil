"""Lossless per-episode content-addressed bytes; never used on the control thread."""
import collections
import hashlib
import os
from pathlib import Path
import re

KEY='__hil_blob_sha256__'


class BlobWriter:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir()
        self.identities=collections.OrderedDict();self.saved={}
        self.pack=(self.root/'data.bin').open('wb')

    def __enter__(self):return self

    def __exit__(self,*args):
        try:
            self.pack.flush();os.fsync(self.pack.fileno())
        finally:self.pack.close()

    def encode(self,value):
        if isinstance(value,bytes) and len(value)>=1024:
            cached=self.identities.get(id(value))
            if cached is not None and cached[0] is value:
                self.identities.move_to_end(id(value));return cached[1]
            digest=hashlib.sha256(value).hexdigest()
            if digest not in self.saved:
                offset=self.pack.tell();self.pack.write(value)
                self.saved[digest]={KEY:digest,'offset':offset,'size':len(value)}
            ref=self.saved[digest]
            self.identities[id(value)]=(value,ref)
            if len(self.identities)>256:self.identities.popitem(last=False)
            return ref
        if isinstance(value,dict):return {k:self.encode(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [self.encode(v) for v in value]
        return value


def decode(value,root):
    if isinstance(value,dict):
        if KEY in value:
            digest=value[KEY]
            if not isinstance(digest,str) or not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('Invalid blob reference')
            if 'offset' in value:
                if not isinstance(value['offset'],int) or value['offset']<0:raise ValueError('Invalid blob offset')
                with (Path(root)/'data.bin').open('rb') as f:
                    f.seek(value['offset']);data=f.read(value['size'])
            else:
                data=(Path(root)/(digest+'.bin')).read_bytes()
            if len(data)!=value['size']:raise ValueError('Incomplete image blob')
            return data
        return {k:decode(v,root) for k,v in value.items()}
    if isinstance(value,list):return [decode(v,root) for v in value]
    return value
