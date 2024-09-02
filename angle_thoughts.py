import numpy as np
from matplotlib import pyplot as plt

close = 26
far = 21

time = 10
angles = np.array([i / 90 * np.pi / 2 for i in range(91)])


def expected_results(hits_per_minute, duration, angle_list):
    expected_counts = hits_per_minute * duration

    def hit_function(angle):
        return expected_counts * np.cos(angle) ** 2

    expected_count_list = np.array([hit_function(angle) for angle in angle_list])
    expected_error_list = np.array([np.sqrt(expected_count) for expected_count in expected_count_list])

    return expected_count_list, expected_error_list


res, err = expected_results(far, time, angles)

plt.plot(angles, res)
plt.fill_between(angles, res + err, res - err, color='red', alpha=0.3)
plt.show()
