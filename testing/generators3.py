
from multiprocessing import Queue
import threading
import abc

class LayerMeasure(abc.ABC):
    def __init__(self,id:int,name:str):
        self.id=id
        self.name=name

    @abc.abstractmethod
    def eval(self,q:Queue,inner_q:Queue):
        pass

    def queue_as_generator(self,q: Queue):
        while True:
            v = q.get()
            if v == None:
                break
            else:
                yield v

class PrintConsumer(LayerMeasure):
    def eval(self,qg,q):
        for i,nada in enumerate(self.queue_as_generator(qg)):
            print(f"inner iteration l={self.id}, i={i}")
            for v in self.queue_as_generator(q):
                assert v==self.id
                print(f"value l={self.id}, i={i}, v={v}")
        print(f"end {self.id}")


class QueueIterator:


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



import time

class SampleFirstPerLayerMeasure(abc.ABC):
    @abc.abstractmethod
    def generate_consumer_for_layer(self,i,name)->LayerMeasure:
        pass

    def run(self,layers,iterations,n_values=4):
        consumers = [self.generate_consumer_for_layer(i,"nombre") for i in range(layers)]
        values = [ [i]*n_values for i in range(layers)]
        queues = [Queue() for i in range(layers)]
        inner_queues = [Queue() for i in range(layers)]

        threads = [threading.Thread(target=c.eval, args=[q,qi]) for c,q,qi in zip(consumers,queues,inner_queues) ]
        for t in threads:
            t.start()

        for i in range(iterations):
            for q in queues:
                q.put(f"Iteracion {i}")
            time.sleep(0.01)
            for i_values in zip(*values):
                time.sleep(0.001)
                for j,v in enumerate(i_values):
                    inner_queues[j].put(v)
            for q in inner_queues:
                q.put(None)
        for q in queues:
            q.put(None)

        for t in threads:
            t.join()
        print("end")


class PrintMeasure(SampleFirstPerLayerMeasure):

    def generate_consumer_for_layer(self,i,name):
        return PrintConsumer(i,name)




if __name__ == '__main__':
    g = PrintMeasure()
    layers=40
    iterations=200
    g.run(layers,iterations)