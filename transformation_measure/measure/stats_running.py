import numpy as np


class RunningMeanSingle:
    def __repr__(self):
        return f"RunningMeanSingle(n={self.n},mean={self.mean()})"

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0

    def clear(self):
        self.n = 0

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.old_m = self.new_m =x
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.old_m = self.new_m

    def mean(self):
        return self.new_m if self.n else 0.0

class RunningMeanSimple:
    def __repr__(self):
        return f"RunningMeanSimple(n={self.n},mean={self.sum().shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.sum = 0

    def update(self, x):
        self.n += 1
        self.sum+=x

    def update_all(self,x:np.ndarray):
        self.sum += x.sum(axis=0)
        self.n += x.shape[0]

    def mean(self):
        return self.sum if self.n > 0 else 0.0

class RunningMean:
    def __repr__(self):
        return f"RunningMean(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.mu = 0

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self.mu = x
        else:
            self.mu = self.mu + (x - self.mu) / self.n

    def update_all(self,x:np.ndarray):
        k=x.shape[0]
        if self.n == 0:
            self.mu = x.mean(axis=0)
            self.n = k
        else:
            self.n += k
            x_sum=x.sum(axis=0)
            self.mu = self.mu + (x_sum - self.mu * k) / self.n

    def mean(self):
        return self.mu if self.n > 0 else 0.0


class RunningMeanAndVariance:

    def __repr__(self):

        return f"RunningMeanAndVariance(n={self.n},mean={self.mean().shape})"

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def var(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return np.sqrt(self.var())
