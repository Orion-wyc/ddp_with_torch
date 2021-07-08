from multiprocessing import Process
import os
import time


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(pId, name):
    info('function f')
    print("rank", pId)
    print('hello', name)


if __name__ == '__main__':
    tic = time.time()
    print(list(map(int, "1,2,3,4".split(','))))
    info('main line')
    pc_list = []
    for i in range(5):
        p = Process(target=f, args=(i, 'bob',))
        p.start()
        pc_list.append(p)
    for p in pc_list:
        p.join()
    toc = time.time()
    print("Elpased Time=", toc - tic)