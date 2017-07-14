
class Losses(object):
    def __init__(self, div):
        self.div = div
        self.clear()

    def clear(self):
        self.loss, self.g, self.d, self.d0, self.d1 = 0.0, 0.0, 0.0, 0.0, 0.0

    def add(self, loss, g, d, d0, d1):
        self.loss += loss / self.div
        self.g += g / self.div
        self.d += d / self.div
        self.d0 += d0 / self.div
        self.d1 += d1 / self.div

    def output(self, s):
        print '%s loss %.2f, g %.2f, d %.2f, adv %.2f, %.2f' \
            % (s, self.loss, self.g, self.d, self.d0, self.d1)
