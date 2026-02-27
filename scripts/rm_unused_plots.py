import os
import argparse

def main(invmass_plot, v1fit_plot):
    """
    Remove plots other than this plot in its directory.
    """
    # Get the directory of the invmass_plot file
    plot_dirs = ['sys_tag_0', 'sys_tag_1', 'sys_tag_2', 'sys_tag_3']
    
    for plot_dir in plot_dirs:
        full_dir = 'plots/' + plot_dir + '/paper_yaml/invmass'
        
        # if does not exist, skip
        if not os.path.exists(full_dir):
            continue
        file_lists = os.listdir(full_dir)
        for file_name in file_lists:
            if file_name != os.path.basename(invmass_plot):
                os.remove(os.path.join(full_dir, file_name))

    for plot_dir in plot_dirs:
        full_dir = 'plots/' + plot_dir + '/paper_yaml/v1fit'

        # if does not exist, skip
        if not os.path.exists(full_dir):
            continue        
        file_lists = os.listdir(full_dir)
        for file_name in file_lists:
            if file_name != os.path.basename(v1fit_plot):
                os.remove(os.path.join(full_dir, file_name))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove unused plot files.")
    parser.add_argument("invmass_plot", type=str, help="Path to the invmass_plot file.")
    parser.add_argument("v1fit_plot", type=str, help="Path to the v1fit_plot file (optional).")
    main(parser.parse_args().invmass_plot, parser.parse_args().v1fit_plot)