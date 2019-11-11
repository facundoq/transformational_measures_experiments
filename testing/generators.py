
from multiprocessing import Queue
import threading

class Consumer():
    def __init__(self):
        pass
    def eval(self,q:Queue):
        for v in queue_as_generator(q):
            print(v)
        print("finishing")


def queue_as_generator(q:Queue):
    while True:
        v = q.get()
        if v == None:
            break
        else:
            yield v



class Generators():



    def run(self):
        c1 = Consumer()
        c2 = Consumer()
        q1, q2 = Queue(), Queue()

        t1 = threading.Thread(target=c1.eval,args=[q1])
        t2 = threading.Thread(target=c2.eval, args=[q2])
        t1.start()
        t2.start()
        g1=[1,3,5,7]
        g2=[2,4,6,8]
        for v1,v2 in zip(g1,g2):
            q1.put(v1)
            q2.put(v2)
        q1.put(None)
        q2.put(None)
        t1.join()
        t2.join()
        print("end")

if __name__ == '__main__':
    g = Generators()
    g.run()