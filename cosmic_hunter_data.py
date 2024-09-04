# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# File lists
filenames_laptop = [
    "Messung1Drausen_10min_2.csv",
    "Messung2Drausen_10min_1_30grad.csv",
    "Messung3Dausen_10min_1_60grad.csv",
    "Messung4drausen_10min_1_60gradccw.csv",
    "Messung5Drausen_10min_1_90Gradccw.csv",
    "Messung6Drausen_10min_90grad.csv",
    "Messung7Drausen_1omin_1_30gradccw.csv",
    "MessungBasement1_10min_3.csv",
    "MessungBasement2_10min_3_30grad.csv",
    "MessungBasement3_10min_3_60grad.csv",
    "MessungBasement4_10min_3_90grad.csv",
    "MessungBasement5_10min_3_30gradcw.csv",
    "MessungBasement6_10min_3_60gradcw.csv",
    "MessungBasement7_10min_3_90gradcw.csv",
    "MessungBasement8_10min_1_Untergrund.csv",
    "MessungBasement8_10min_1_Untergrund2.csv"
]

file_names_outside = [
    "Messung5Drausen_10min_1_90Gradccw.csv",
    "Messung4drausen_10min_1_60gradccw.csv",
    "Messung7Drausen_1omin_1_30gradccw.csv",
    "Messung1Drausen_10min_2.csv",
    "Messung2Drausen_10min_1_30grad.csv",
    "Messung3Dausen_10min_1_60grad.csv",
    "Messung6Drausen_10min_90grad.csv",
]

angle_range_whole = np.array([-90, -60, -30, 0, 30, 60, 90])


def get_measurement_data(folder_name: str, file_name: str, print_output: bool = False):
    file_path = f'data/{folder_name}/{file_name}'

    df = pd.read_csv(file_path)

    acquisition_time = np.sum(df['Acquisition Time'].to_numpy())
    coincidences = np.sum(df['Coincidences'].to_numpy())
    coincidence_error = np.sqrt(coincidences)
    hits_per_second = coincidences / acquisition_time
    hits_per_second_error = coincidence_error / acquisition_time

    if print_output:
        print(f'file name: {file_name}')
        print(f'Total acquisition time: {acquisition_time}')
        print(f'Total coincidences: {coincidences} ± {round(coincidence_error, 3)}')
        print(f'Hits per second: {hits_per_second} ± {round(hits_per_second_error, 3)}')
        print(f'--> {round(100 * coincidence_error / coincidences, 2)}% relative error')
        print('\n')

    return acquisition_time, (coincidences, coincidence_error), (hits_per_second, hits_per_second_error)


laptop_data = []

for name in file_names_outside:
    new_data = get_measurement_data('Cosmic_hunter_data_laptop', name, True)
    laptop_data.append(new_data)

hits_per_second_data = [ld[2][0] for ld in laptop_data]
hits_per_second_error_data = [ld[2][1] for ld in laptop_data]
lin_angles = np.linspace(-120, 120, 100)
theoretical_counts = max(hits_per_second_data) * np.cos(lin_angles * np.pi / 180) ** 2

plt.figure(figsize=(12, 5))
plt.title('Angular dependence of muon detections', size=16)
plt.xlabel(r'Angle $\varphi$ in [°]', size=13)
plt.ylabel('Coincidence counts in [1/s]', size=13)

plt.scatter(angle_range_whole, hits_per_second_data, label='measured data', marker='x')
plt.plot(lin_angles, theoretical_counts, label='expectation', c='orange')
plt.errorbar(angle_range_whole, hits_per_second_data, hits_per_second_error_data, fmt='none', capsize=5, ecolor='black')

plt.xlim(-105, 105)
plt.tight_layout()
plt.legend()
plt.savefig('angular_dependence_of_muon_detections.png', dpi=200)
plt.show()
