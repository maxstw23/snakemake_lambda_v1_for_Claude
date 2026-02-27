import numpy as np
import os

def main():
    print(os.getcwd())
    data_dict = {}
    energy_correspondence = {
        '27': '27GeV',
        '19': '19.6GeV',
        '17': '17.3GeV',
        '14': '14.6GeV',
        '11': '11.5GeV',
        '9': '9.2GeV',
        '7': '7.7GeV'
    }
    for energy in ['27', '19', '17', '14', '11' ,'9', '7']:
        e_key = energy_correspondence[energy]
        data_dict[e_key] = {}
        for particle in ['pions', 'kaons', 'protons']:
            data_dict[e_key][particle] = {}
            filename = f'./Etabins_20_Coalescence/Etabins_20_Coalescence/txtfiles/{energy}GeV_{particle}_datapoints.txt'
            
            with open(filename, 'r') as file:
                data = file.readlines()
            # look for the line with "v1slopes_cubic_combinedcent"
            for line in data:
                if 'v1_vCent' in line:
                    if 'p_linear[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['pos_linear'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'p_linear_err[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['pos_linear_err'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'p_linear_systematics[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['pos_linear_systematics'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'p_cubic[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['pos_cubic'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'p_cubic_err[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['pos_cubic_err'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'p_cubic_systematics[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['pos_cubic_systematics'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'n_linear[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['neg_linear'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'n_linear_err[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['neg_linear_err'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'n_linear_systematics[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['neg_linear_systematics'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'n_cubic[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['neg_cubic'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'n_cubic_err[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['neg_cubic_err'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'n_cubic_systematics[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['neg_cubic_systematics'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'delta' in line and 'linear[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['delta_linear'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'delta' in line and 'linear_err[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['delta_linear_err'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'delta' in line and 'linear_systematics[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['delta_linear_systematics'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'delta' in line and 'cubic[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['delta_cubic'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'delta' in line and 'cubic_err[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['delta_cubic_err'] = np.array([float(valstr) for valstr in valstr_list])
                    elif 'delta' in line and 'cubic_systematics[9]' in line:
                        valstr_list = line.split('{')[1].split('}')[0].split(',')
                        data_dict[e_key][particle]['delta_cubic_systematics'] = np.array([float(valstr) for valstr in valstr_list])

                if 'v1slopes_linear_combinedcent' in line:
                    # extract the values after the label, in between {}
                    if 'pos[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_linear'] = {}
                            data_dict[e_key][particle][f'{cent}_linear']['pos'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'pos_err[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_linear']['pos_err'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'pos_systematics[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_linear']['pos_systematics'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'neg[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_linear']['neg'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'neg_err[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_linear']['neg_err'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'neg_systematics[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_linear']['neg_systematics'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'deltav1[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_linear']['delta'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'deltav1_err[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_linear']['delta_err'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'deltav1_systematics[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_linear']['delta_systematics'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                elif 'v1slopes_cubic_combinedcent' in line:
                    if 'pos[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_cubic'] = {}
                            data_dict[e_key][particle][f'{cent}_cubic']['pos'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'pos_err[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_cubic']['pos_err'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'pos_systematics[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_cubic']['pos_systematics'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'neg[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_cubic']['neg'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'neg_err[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_cubic']['neg_err'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'neg_systematics[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_cubic']['neg_systematics'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'deltav1[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_cubic']['delta'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'deltav1_err[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_cubic']['delta_err'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())
                    elif 'deltav1_systematics[4]' in line:
                        for i, cent in enumerate(['010','1040', '4080', '5080']):
                            data_dict[e_key][particle][f'{cent}_cubic']['delta_systematics'] = float(line.split('{')[1].split('}')[0].split(',')[i].strip())

    # print the data_dict as if it were created by a python script
    with open('scripts/pikp_merged.py', 'w') as f:
        f.write('import numpy as np\n\n')
        f.write("""class PikpMergedSlope:
    def __init__(self):
        self.data = {\n""")         
        
        for energy, particles in data_dict.items():
            f.write(f"\t\t\t'{energy}': {{\n")
            for particle, centroids in particles.items():
                f.write(f"\t\t\t\t'{particle}': {{\n")
                for centroid, values in centroids.items():
                    if isinstance(values, np.ndarray):
                        f.write(f"\t\t\t\t\t'{centroid}': np.array({values.tolist()}),\n")
                    else:
                        f.write(f"\t\t\t\t\t'{centroid}': {values},\n")
                f.write("\t\t\t\t},\n")
            f.write("\t\t\t},\n")
        f.write("\t\t}\n\n")

        f.write("""
    def get_data(self):
        return self.data""")
    # print("self.data = {")
    # for energy, particles in data_dict.items():
    #     print(f"    '{energy}': {{")
    #     for particle, centroids in particles.items():
    #         print(f"        '{particle}': {{")
    #         for centroid, values in centroids.items():
    #             if isinstance(values, np.ndarray):
    #                 print(f"            '{centroid}': np.array({values.tolist()}),")
    #             else:
    #                 print(f"            '{centroid}': {values},")
    #         print("        },")
    #     print("    },")
    # print("}")
            

if __name__ == "__main__":
    main()
   