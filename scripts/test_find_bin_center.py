import numpy as np
from find_bin_center import BinCenterFinder

def main():
    # Example bin edges and heights
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    bin_heights = np.array([5011, 4010, 3020, 1400, 801])

    # Create an instance of BinCenterFinder
    finder = BinCenterFinder(bin_edges, bin_heights, 3)

    # Find the bin centers
    bin_centers = finder.find_bin_centers()
    finder.visualize()

    # Print the results
    print("Bin Centers:", bin_centers)


if __name__ == "__main__":
    main()