import numpy as np


def add_helper(v, c, s):
    """
    Expect reshaped input.
    :param v: Values. Should be of shape (k, n) where k runs over
              profiles to add and n is the number of bin, same below.
    :param c: Counts
    :param s: Standard deviation of the y values.
    :return: Added v, c, and s
    """
    cc = np.sum(c, axis=0)
    vv = np.divide(np.sum(v * c, axis=0), cc, out=np.zeros(np.shape(cc)), where=cc != 0)
    E = np.sum(c * (s ** 2 + v ** 2), axis=0)
    ss = (np.divide(E, cc, out=np.zeros(np.shape(cc)), where=cc != 0) - vv ** 2) ** 0.5
    return vv, cc, ss


class SimpleProfile:
    def __init__(self, values, counts, std_devs, edges_or_bin_centers, use_edges=True):
        if len(values) != len(counts):
            raise ValueError('Values and counts not the same length!')
        if len(values) != len(std_devs):
            raise ValueError('Values and std_devs not the same length!')
        if use_edges:
            if len(values) != len(edges_or_bin_centers) - 1:
                raise ValueError('Edges should have n+1 dimension!')
        else:
            if len(values) != len(edges_or_bin_centers):
                raise ValueError('Bin centers should have n dimension!')

        self.v = values
        self.c = counts
        self.s = std_devs
        if use_edges:
            self.e = edges_or_bin_centers
            self.bc = 0.5 * (self.e[:-1] + self.e[1:])
        else:
            self.bc = edges_or_bin_centers
            width = edges_or_bin_centers[1] - edges_or_bin_centers[0]
            self.e = np.linspace(edges_or_bin_centers[0] - width / 2,
                                 edges_or_bin_centers[-1] + width / 2,
                                 len(edges_or_bin_centers) + 1)

    def values(self):
        return self.v

    def counts(self):
        return self.c

    def errors(self):
        sumwy2 = np.sum(self.c*self.v**2)
        sumwy  = np.sum(self.c*self.v)
        sumw   = np.sum(self.c)
        if sumw == 0:
            sumw = 1
        global_mean = sumwy / sumw

        cond = (sumwy2 != 0) & (self.c < 5)
        test = np.where(cond, self.s * self.s * self.c / sumwy2, 1)
        zero_cond = (test < 1e-4) | (self.s <= 0) | (self.c <= 0)
        err = np.divide(self.s, self.c ** 0.5, out=np.zeros(np.shape(self.s)), where=np.logical_not(zero_cond))
        zero_ind = np.where(zero_cond)
        err[zero_ind] = 2*(abs(sumwy2/sumw - global_mean**2))**0.5
        return err

    def edges(self):
        return self.e

    def bin_centers(self):
        return self.bc

    def Rebin(self, nrebin):
        if nrebin == 1:
            return

        nbin = len(self.v)
        if nbin % nrebin != 0:
            raise ValueError('nrebin must divide the number of bins!')
        old_c = self.c.copy()
        self.v, self.c, self.s = add_helper(self.v.reshape(nbin // nrebin, nrebin).T,
                                            self.c.reshape(nbin // nrebin, nrebin).T,
                                            self.s.reshape(nbin // nrebin, nrebin).T)
        self.e = np.linspace(self.e[0], self.e[-1], nbin // nrebin + 1)

        # new bin centers should be weighted average of the old ones based on counts
        self.e = np.arange(self.e[0], self.e[-1], nrebin * (self.e[1] - self.e[0]))
        self.bc = np.average(self.bc.reshape(-1, nrebin), weights=old_c.reshape(-1, nrebin), axis=1)

    def __add__(self, other):
        vi = np.vstack((self.v, other.v))
        ci = np.vstack((self.c, other.c))
        si = np.vstack((self.s, other.s))
        vo, co, so = add_helper(vi, ci, si)
        return SimpleProfile(vo, co, so, self.e)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


if __name__ == '__main__':
    print('Testing SimpleProfile class...')