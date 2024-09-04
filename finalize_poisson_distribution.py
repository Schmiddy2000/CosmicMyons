# Imports
import json
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import poisson, chisquare


def read_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    remaining_times = data.get('remaining_times', [])
    coincidence_counts = data.get('coincidence_counts', [])

    return remaining_times, coincidence_counts


my_times, my_counts = read_from_json('test_data_1.json')

count_histo_list = []

first_time = my_times[0]


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
            if abs(value_diff) < 20:# and value_diff > 0:
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


# def plot_histograms(one_diff, two_diff):
#     plt.figure(figsize=(12, 6))
#
#     # Histogram for one identical time value differences
#     plt.subplot(1, 2, 1)
#     plt.hist(one_diff, bins=max(one_diff) + 1, color='blue', edgecolor='black')
#     plt.title('Histogram of Differences (One Identical Time Value)')
#     plt.xlabel('Difference')
#     plt.ylabel('Frequency')
#
#     # Histogram for two identical time value differences
#     plt.subplot(1, 2, 2)
#     plt.hist(two_diff, bins=max(two_diff) - 1, color='green', edgecolor='black')
#     plt.title('Histogram of Differences (Two Identical Time Values)')
#     plt.xlabel('Difference')
#     plt.ylabel('Frequency')
#
#     # Show the histograms
#     plt.tight_layout()
#     plt.show()


def plot_histograms(one_diff, two_diff):
    plt.figure(figsize=(12, 6))

    # Unique values for bins
    unique_one_diff = np.unique(one_diff)
    unique_two_diff = np.unique(two_diff)

    # Histogram for one identical time value differences
    plt.subplot(1, 2, 1)
    plt.hist(one_diff, bins=unique_one_diff - 0.5, align='mid', rwidth=0.9, color='blue', edgecolor='black')
    plt.title('Histogram of Differences (One Identical Time Value)')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')

    # Histogram for two identical time value differences
    plt.subplot(1, 2, 2)
    plt.hist(two_diff, bins=unique_two_diff - 0.5, align='mid', rwidth=0.9, color='green', edgecolor='black')
    plt.title('Histogram of Differences (Two Identical Time Values)')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')

    # Show the histograms
    plt.tight_layout()
    plt.show()


# Example usage
# plot_histograms(ones, twos)

def plot_histogram_with_poisson_fit(two_diff):
    # Calculate the mean (lambda) for the Poisson distribution
    lambda_estimate = np.mean(two_diff)

    # Create a histogram
    plt.figure(figsize=(8, 6))
    count, bins, ignored = plt.hist(two_diff, bins=np.unique(two_diff) - 0.5, align='mid', rwidth=0.9, color='green',
                                    edgecolor='black', density=True)

    # Generate Poisson PMF based on the estimated lambda
    x = np.arange(min(two_diff), max(two_diff) + 1)
    poisson_pmf = poisson.pmf(x, lambda_estimate)

    # Plot the Poisson PMF
    plt.plot(x, poisson_pmf, 'r-', lw=2, label=f'Poisson Fit (λ={lambda_estimate:.2f})')
    plt.title('Histogram of Differences (Two Identical Time Values) with Poisson Fit')
    plt.xlabel('Difference')
    plt.ylabel('Frequency (normalized)')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


# plot_histogram_with_poisson_fit(twos)


def plot_histogram_with_continuous_poisson_fit(two_diff):
    # Calculate the mean (lambda) for the Poisson distribution
    lambda_estimate = np.mean(two_diff)

    # Create a histogram
    plt.figure(figsize=(8, 6))
    count, bins, ignored = plt.hist(two_diff, bins=np.unique(two_diff) - 0.5, align='mid', rwidth=0.9, color='green',
                                    edgecolor='black', density=True)

    # Generate Poisson PMF based on the estimated lambda
    x = np.linspace(min(two_diff) - 0.5, max(two_diff) + 0.5, 1000)
    poisson_pmf = poisson.pmf(np.floor(x + 0.5), lambda_estimate)

    # Plot the continuous Poisson fit as a line
    plt.plot(x, poisson_pmf, 'r-', lw=2, label=f'Poisson Fit (λ={lambda_estimate:.2f})')
    plt.title('Histogram of Differences (Two Identical Time Values) with Continuous Poisson Fit')
    plt.xlabel('Difference')
    plt.ylabel('Frequency (normalized)')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


# plot_histogram_with_continuous_poisson_fit(twos)


def calculate_lambda_and_uncertainty(two_diff):
    # Estimate the lambda (mean)
    lambda_estimate = np.mean(two_diff)

    # Calculate the standard error of the lambda estimate
    n = len(two_diff)
    lambda_uncertainty = np.sqrt(lambda_estimate / n)

    return lambda_estimate, lambda_uncertainty


def plot_histogram_with_continuous_poisson_fit(two_diff):
    # Calculate lambda and its uncertainty
    lambda_estimate, lambda_uncertainty = calculate_lambda_and_uncertainty(two_diff)

    # Print lambda and its uncertainty
    print(f"Estimated λ: {lambda_estimate:.2f} ± {lambda_uncertainty:.2f}")

    # Create a histogram
    plt.figure(figsize=(8, 6))
    count, bins, ignored = plt.hist(two_diff, bins=np.unique(two_diff) - 0.5, align='mid', rwidth=0.9, color='green',
                                    edgecolor='black', density=True)

    # Generate Poisson PMF based on the estimated lambda
    x = np.linspace(min(two_diff) - 0.5, max(two_diff) + 0.5, 1000)
    poisson_pmf = poisson.pmf(np.floor(x + 0.5), lambda_estimate)

    # Plot the continuous Poisson fit as a line
    plt.plot(x, poisson_pmf, 'r-', lw=2, label=f'Poisson Fit (λ={lambda_estimate:.2f} ± {lambda_uncertainty:.2f})')
    plt.title('Histogram of Differences (Two Identical Time Values) with Continuous Poisson Fit')
    plt.xlabel('Difference')
    plt.ylabel('Frequency (normalized)')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


# plot_histogram_with_continuous_poisson_fit(twos)

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
    plt.savefig('histogram_coincidence_counts_1_and_2', dpi=175)
    plt.show()


plot_histograms_with_poisson_and_chisquare(ones, twos)
