"""Cross-process locks for atomic, resumable per-run experiment scheduling."""
from contextlib import contextmanager
from hashlib import sha256
import os
import time

from fedperg.paths import LOGS

if os.name=='nt':
    import msvcrt
else:
    import fcntl


@contextmanager
def record_lock(name):
    folder=LOGS/'run_locks'
    folder.mkdir(exist_ok=True)
    path=folder/(sha256(name.encode()).hexdigest()+'.lock')
    with path.open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        while True:
            try:
                handle.seek(0)
                if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:time.sleep(.5)
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
