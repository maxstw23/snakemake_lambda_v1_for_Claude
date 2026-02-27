import numpy as np

class BinCenterFinder:
    def __init__(self, bin_edges, bin_heights, fit_order=3, bin_devision=1000):
        if len(bin_edges) != len(bin_heights) + 1:
            raise ValueError("Length of bin_edges must be one more than length of bin_heights.")
        self.bin_edges = bin_edges
        self.bin_centers_visual = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.bin_heights = bin_heights
        self.fit_order = fit_order
        self.bin_devision = bin_devision

    def find_bin_centers(self):
        """
        Iteratively fit the bin heights vs bin centers to find the true "centroid" of the bins.
        """
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        old_centers = np.ones_like(bin_centers) * np.inf
        while not np.allclose(old_centers, bin_centers, atol=1e-5):
            old_centers = bin_centers
            fit = np.polynomial.Polynomial.fit(bin_centers, self.bin_heights, self.fit_order, domain=[self.bin_edges[0], self.bin_edges[-1]])
            xx, yy = fit.linspace(n=self.bin_devision * len(bin_centers))
            bin_centers = np.sum(np.reshape(xx*yy, (len(bin_centers), self.bin_devision)), axis=1) / np.sum(np.reshape(yy, (len(bin_centers), self.bin_devision)), axis=1)
            
        return bin_centers
    
    def visualize(self):
        """
        Visualize with animation of the fitting process.
        """

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots()
        ax.set_xlim(self.bin_edges[0], self.bin_edges[-1])
        ax.set_ylim(min(self.bin_heights) - 100, max(self.bin_heights) + 100)

        line, = ax.plot([], [], 'r-', label='Fit')
        scatter = ax.scatter(self.bin_centers_visual, self.bin_heights, color='blue', label='Bin Centers')
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
        bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])     

        def update(frame):

            fit = np.polynomial.Polynomial.fit(self.bin_centers_visual, self.bin_heights, self.fit_order, domain=[self.bin_edges[0], self.bin_edges[-1]])
            xx, yy = fit.linspace(n=self.bin_devision * len(self.bin_centers_visual))
            line.set_data(xx, yy)
            print(f"Frame {frame}: Current bin centers: {self.bin_centers_visual}")
            scatter.set_offsets(np.stack((self.bin_centers_visual, self.bin_heights)).T)
            time_text.set_text(f'Frame: {frame}')

            # update bin centers
            self.bin_centers_visual = np.sum(np.reshape(xx*yy, (len(self.bin_centers_visual), self.bin_devision)), axis=1) / np.sum(np.reshape(yy, (len(self.bin_centers_visual), self.bin_devision)), axis=1)
            return (line, scatter, time_text)

        ani = FuncAnimation(fig, update, frames=10, blit=True, repeat=True, interval=500)
        # show frame number
        # plt.title(f'Frame: {ani._framenumber}')
        plt.legend()
        plt.show()


            
