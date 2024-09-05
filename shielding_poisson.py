# Imports
import json
from os import times

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import poisson, chisquare, linregress
from scipy.optimize import curve_fit


def read_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    remaining_times = data.get('remaining_times', [])
    coincidence_counts = data.get('coincidence_counts', [])

    return remaining_times, coincidence_counts


my_times, my_counts = read_from_json('shielding_measurement_4.json')


def calculate_lambda_and_uncertainty(two_diff):
    # Estimate the lambda (mean)
    lambda_estimate = np.mean(two_diff)

    # Calculate the standard error of the lambda estimate
    n = len(two_diff)
    lambda_uncertainty = np.sqrt(lambda_estimate / n)

    return lambda_estimate, lambda_uncertainty


print(len(my_times), len(my_counts))
print(my_times)
print(my_counts)


def extract_value_counts(times, counts):
    one_identical_value_diff = []
    two_identical_value_diff = []

    i = 0
    while i < len(times) - 3:
        # Track the number of identical consecutive time values
        count_identical = 1

        while i + count_identical < len(times) and times[i] == times[i + count_identical]:
            count_identical += 1

        if count_identical == 1:
            # Case for one identical time value
            value_diff = counts[i + 1] - counts[i]
            if abs(value_diff) < 20 and value_diff > 0:
                one_identical_value_diff.append(value_diff)
        elif count_identical == 2:
            # Case for two identical time values
            value_diff = counts[i + 2] - counts[i]
            if abs(value_diff) < 20 and value_diff > 0:
                two_identical_value_diff.append(value_diff)
        else:
            # More than two identical time values, skip these values
            print(f"Skipping {count_identical} identical time values starting at index {i}")

        # Move the index to the next new time value
        i += count_identical

    return one_identical_value_diff, two_identical_value_diff


ones, twos = extract_value_counts(my_times, my_counts)

print(ones)
print(twos)

print(len(ones))
print(len(twos))

measurement_counts = [375, 397, 347, 381, 386, 1499]
measurement_times = [167, 184, 184, 183, 205, 773]

lambda_list = []
lambda_uncertainty_list = []

for i in range(1, 7):
    t, c = read_from_json(f'shielding_measurement_{i}.json')
    one, two = extract_value_counts(t, c)

    l, delta_l = calculate_lambda_and_uncertainty(two)
    l_2, delta_l_2 = calculate_lambda_and_uncertainty(one)
    # lambda_list.append(l / 2)
    # lambda_list.append(l_2)
    lambda_list.append((l / 2 + l_2) / 2)
    combined_lambda_uncertainty = 1 / 2 * np.sqrt(delta_l ** 2 / 4 + delta_l_2 ** 2)
    lambda_uncertainty_list.append(combined_lambda_uncertainty)


lambda_list = [measurement_counts[i] / measurement_times[i] for i in range(len(measurement_counts))]
lambda_uncertainty_list = [np.sqrt(measurement_counts[i]) / measurement_times[i] for i in range(len(measurement_counts))]


# Define the exponential function to fit
def exponential_func(x, A, B):
    return A * np.exp(B * x)


# Data points
x_data = np.array([0, 0.5, 0.8, 1.3, 4.5, 9])
y_data = np.array(lambda_list)
y_err = np.array(lambda_uncertainty_list)

# Perform linear regression using scipy's linregress
slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

# Generate linear fit values
y_pred_linear = slope * x_data + intercept

# Linear fit confidence band: using standard error
y_pred_std_err = np.std(y_data - y_pred_linear)

# Perform exponential fit
popt_exp, pcov_exp = curve_fit(exponential_func, x_data, y_data, sigma=y_err, absolute_sigma=True)
A, B = popt_exp
exp_fit = exponential_func(x_data, A, B)

# Get standard deviations for parameters from covariance matrix (for error bands)
perr_exp = np.sqrt(np.diag(pcov_exp))

# Generate smooth curve for plotting the exponential fit
x_smooth = np.linspace(min(x_data), max(x_data), 100)
y_smooth_exp = exponential_func(x_smooth, *popt_exp)

plt.figure(figsize=(12, 5))

# Plot data points with error bars
plt.scatter(x_data, y_data, label='Data points', color='blue')
plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', color='blue', capsize=4)

# Plot linear fit with confidence band
plt.plot(x_data, y_pred_linear, color='red', label=f'Linear fit (slope = {slope:.2f} ± {std_err:.2f})')
plt.fill_between(x_data, y_pred_linear - y_pred_std_err, y_pred_linear + y_pred_std_err,
                 color='red', alpha=0.2, ls='--',  label='Linear 1-sigma')

# Plot exponential fit
plt.plot(x_smooth, y_smooth_exp, color='green', label=f'Exponential Fit: A={A:.2f}, B={B:.2f}')
plt.fill_between(x_smooth, exponential_func(x_smooth, A - perr_exp[0], B - perr_exp[1]),
                 exponential_func(x_smooth, A + perr_exp[0], B + perr_exp[1]),
                 color='green', alpha=0.2, ls='--', label='Exponential 1-sigma')

# Add labels and legend
plt.xlabel('Lead shielding thickness in [cm]')
plt.ylabel('Coincidence counts in [1/s]')
plt.legend()
plt.title('Linear and Exponential Fits with Confidence Intervals')
plt.tight_layout()
plt.savefig('shielding_impact.png', dpi=200)
plt.show()

# x_values = np.array([0, 0.5, 0.8, 1.3, 4.5, 9])
#
# # Perform linear regression
# slope, intercept, r_value, p_value, std_err = linregress(x_values, lambda_list)
#
#
# # Define a function for the regression line
# def regression_line(x):
#     return slope * x + intercept
#
#
# # Generate x values for the regression line (for smoother line)
# x_regression = np.linspace(min(x_values) - 1, max(x_values) + 1, 100)
# y_regression = regression_line(x_regression)
#
# # Calculate residuals and standard deviation of the residuals
# residuals = np.array(lambda_list) - regression_line(x_values)
# residual_std_error = np.std(residuals)
#
# # Plotting
# plt.scatter(x_values, lambda_list, color='blue', label='Data points')
# plt.errorbar(x_values, lambda_list, yerr=lambda_uncertainty_list, fmt='none', ecolor='black', capsize=4)
#
# # Plot regression line
# plt.plot(x_regression, y_regression, color='black', ls='--', label=f'Linear fit (slope = {slope:.2f} ± {std_err:.2f})')
#
# # Add 1-sigma confidence band
# plt.fill_between(x_regression,
#                  y_regression - residual_std_error,
#                  y_regression + residual_std_error,
#                  color='red', ls='--', alpha=0.2, label='1-sigma confidence')
#
# # Adding labels and legend
# plt.xlabel('Lead shielding thickness in [cm]')
# plt.ylabel('Lambda Value')
# plt.xlim(min(x_values) - 1, max(x_values) + 1)
# plt.legend()
# plt.show()

# ---

# def extract_value_counts(times, counts):
#     one_identical_value_diff = []
#     two_identical_value_diff = []
#     skipped_values = []  # To track values with more than two identical entries
#
#     identical_times_counts = []
#     current_identical_time_count = 0
#
#     i = 0
#     current_time_value = times[0]
#
#     for i, time in enumerate(times):
#         print(time)
#         if time == current_time_value:
#             current_identical_time_count += 1
#
#         elif time > current_time_value:
#             identical_times_counts.append(current_identical_time_count)
#             current_identical_time_count = 0
#             current_time_value = time
#
#         else:
#             # This is the error case. How can I best handle this?
#             print(f'error: {i}, {times[i-1]}, {time}')
#
#     print(identical_times_counts)
#
#     return None


# extract_value_counts(my_times, my_counts)

# def extract_value_counts(times, counts):
#     one_identical_value_diff = []
#     two_identical_value_diff = []
#     skipped_values = []  # To track values with more than two identical entries
#
#     i = 0
#     while i < len(times) - 3:
#         # Track the number of identical consecutive time values
#         count_identical = 1
#
#         while i + count_identical < len(times) and times[i] == times[i + count_identical]:
#             count_identical += 1
#
#         if count_identical == 1:
#             # Case for one identical time value
#             value_diff = counts[i + 1] - counts[i]
#             if abs(value_diff) < 20:
#                 one_identical_value_diff.append(value_diff)
#         elif count_identical == 2:
#             # Case for two identical time values
#             value_diff = counts[i + 2] - counts[i]
#             if abs(value_diff) < 20 and value_diff > 0:
#                 two_identical_value_diff.append(value_diff)
#         else:
#             # Case for more than two identical values
#             print(f"Skipping {count_identical} identical time values starting at index {i}")
#             skipped_values.append(times[i:i + count_identical])
#
#         # Move the index to the next new time value
#         i += count_identical
#
#     return one_identical_value_diff, two_identical_value_diff, skipped_values


# Call the function and print results
# ones, twos, skipped = extract_value_counts(my_times, my_counts)
#
# print("Ones:", ones)
# print("Twos:", twos)
# print("Skipped:", skipped)
#
# print("Length of ones:", len(ones))
# print("Length of twos:", len(twos))


def calculate_lambda_and_uncertainty(two_diff):
    # Estimate the lambda (mean)
    lambda_estimate = np.mean(two_diff)

    # Calculate the standard error of the lambda estimate
    n = len(two_diff)
    lambda_uncertainty = np.sqrt(lambda_estimate / n)

    return lambda_estimate, lambda_uncertainty


def chi_square_test(observed, expected):
    # Scale expected frequencies to match the sum of observed frequencies
    expected_scaled = expected * (np.sum(observed) / np.sum(expected))
    chi_square_statistic, p_value = chisquare(f_obs=observed, f_exp=expected_scaled)
    return chi_square_statistic, p_value


def plot_histograms_with_poisson_and_chisquare(one_diff, two_diff):
    # Calculate lambda and uncertainties
    lambda_one, lambda_one_uncertainty = calculate_lambda_and_uncertainty(one_diff)
    lambda_two, lambda_two_uncertainty = calculate_lambda_and_uncertainty(two_diff)

    # Define the bins
    bins_one = np.arange(min(one_diff), max(one_diff) + 2) - 0.5
    bins_two = np.arange(min(two_diff), max(two_diff) + 2) - 0.5

    # Calculate observed frequencies
    observed_one, _ = np.histogram(one_diff, bins=bins_one)
    observed_two, _ = np.histogram(two_diff, bins=bins_two)

    # Calculate expected frequencies
    expected_one = poisson.pmf(np.arange(min(one_diff), max(one_diff) + 1), lambda_one) * len(one_diff)
    expected_two = poisson.pmf(np.arange(min(two_diff), max(two_diff) + 1), lambda_two) * len(two_diff)

    # Perform Chi-Square tests
    chi_square_one, p_value_one = chi_square_test(observed_one, expected_one)
    chi_square_two, p_value_two = chi_square_test(observed_two, expected_two)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for 'ones'
    axs[0].hist(one_diff, bins=bins_one, align='mid', rwidth=0.9, color='blue', edgecolor='black', density=True)
    x_one = np.linspace(min(one_diff) - 0.5, max(one_diff) + 0.5, 1000)
    poisson_pmf_one = poisson.pmf(np.floor(x_one + 0.5), lambda_one)
    axs[0].plot(x_one, poisson_pmf_one, 'r-', lw=2,
                label=f'Poisson Fit (λ={lambda_one:.2f} ± {lambda_one_uncertainty:.2f})\n$\chi^2$={chi_square_one:.2f}, p={p_value_one:.3f}')
    axs[0].set_title('Coincidence counts (1-second)')
    axs[0].set_xlabel('Counts')
    axs[0].set_ylabel('Frequency (normalized)')
    axs[0].legend()

    # Plot for 'twos'
    axs[1].hist(two_diff, bins=bins_two, align='mid', rwidth=0.9, color='green', edgecolor='black', density=True)
    x_two = np.linspace(min(two_diff) - 0.5, max(two_diff) + 0.5, 1000)
    poisson_pmf_two = poisson.pmf(np.floor(x_two + 0.5), lambda_two)
    axs[1].plot(x_two, poisson_pmf_two, 'r-', lw=2,
                label=f'Poisson Fit (λ={lambda_two:.2f} ± {lambda_two_uncertainty:.2f})\n$\chi^2$={chi_square_two:.2f}, p={p_value_two:.3f}')
    axs[1].set_title('Coincidence counts (2-second)')
    axs[1].set_xlabel('Counts')
    axs[1].set_ylabel('Frequency (normalized)')
    axs[1].legend()

    plt.tight_layout()
    # plt.savefig('histogram_coincidence_counts_1_and_2', dpi=175)
    plt.show()


plot_histograms_with_poisson_and_chisquare(ones, twos)
