
from multiprocessing import Queue
import threading

class Consumer():
    def __init__(self,index):
        self.index=index
    def eval(self,qg:Queue,q):
        for i,t in enumerate(qg):
            for v in q:
                print(f"value {self.index}_{i}_{v}")




class QueueIterator:

    """Iterator that counts upward forever."""

    def __init__(self, q:Queue):
        self.q = q

    def __iter__(self):
        return self

    def __next__(self):
        v=self.q.get()
        if v == None:
            raise StopIteration
        else:
            return v


class Generators():
    def run(self):
        c1 = Consumer(0)
        c2 = Consumer(1)
        g1=[1,3,5,7]
        g2=[2,4,6,8]
        n=3
        q1 = Queue()
        qg1 = QueueIterator(q1)
        qi1 = Queue()
        qig1 = QueueIterator(qi1)

        q2 = Queue()
        qg2 = QueueIterator(q2)
        qi2 = Queue()
        qig2 = QueueIterator(qi2)

        t1 = threading.Thread(target=c1.eval, args=[qg1,qig1])
        t2 = threading.Thread(target=c2.eval, args=[qg2,qig2])
        t1.start()
        t2.start()
        for i in range(n):
            print(f"iteration {i}")
            q1.put(1)
            q2.put(2)
            for v1,v2 in zip(g1,g2):
                qi1.put(v1)
                qi2.put(v2)
            qi1.put(None)
            qi2.put(None)
        q1.put(None)
        q2.put(None)

        t1.join()
        t2.join()
        print("end")


if __name__ == '__main__':
    g = Generators()
    g.run()