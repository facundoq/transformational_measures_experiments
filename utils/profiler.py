from datetime import datetime

def get_epochtime_ms():
    return round(datetime.utcnow().timestamp() * 1000)

class Clock:
    def __init__(self):
        self.time=self.now()
    def now(self):
        return get_epochtime_ms()
    def update(self):
        elapsed=self.elapsed()
        self.time=self.now()
        return elapsed
    def elapsed(self):
        return self.now()-self.time


class Profiler:

    def __init__(self):
        self.reset()
    def event(self, name):
        self.measures.append(get_epochtime_ms())
        self.names.append(name)


    def human_readable_time(self, t:int)->str:
        # alternatively, use timedelta(milliseconds=t) and

        divide=[1000,60,60]
        values=[]
        for d in divide:
            values.append(t % d)
            t = t//d
        values.append(t)
        ms,hms=values[0],reversed(values[1:])
        hms_string= ":".join([f"{v:02}" for v in hms])
        return f"{hms_string}.{ms}"

    def summary(self, human=False):
        if len(self.measures)>1:
            # deltas=[j-i for i, j in zip()]
            vals=zip(self.names[:-1], self.names[1:],self.measures[:-1],self.measures[1:])
            time_format=lambda t: self.human_readable_time(t) if human else f"{t}ms"
            tags = [f"{n1} to {n2}: {time_format(t2-t1)}" for n1,n2,t1,t2 in vals]
            return "\n".join(tags)
        elif len(self.measures)==1:
            return f"One measure ({self.name[0]})."
        else:
            return "No measures."

    def reset(self):
        self.measures=[]
        self.names=[]

