
class Accumulator(object):
    def __init__(self, div, names):
        self.div = div
        self.names = names
        self.n = len(self.names)
        self.values = [0] * self.n

    def clear(self):
        self.values = [0] * self.n

    def add(self, values):
        for i in range(self.n):
            self.values[i] += values[i] / self.div

    def output(self, s=''):
        if s:
            s += ' '
        for i in range(self.n):
            s += '%s %.2f' % (self.names[i], self.values[i])
            if i < self.n-1:
                s += ', '
        print(s)
