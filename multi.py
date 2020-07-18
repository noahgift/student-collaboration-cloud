import sokoban
from multiprocessing import Process

def f():
    sokoban.main()

if __name__ == '__main__':
    p = Process(target=f)
    p.start()
    p.join()

