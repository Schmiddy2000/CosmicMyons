# Imports
import json
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import poisson


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


plot_histogram_with_continuous_poisson_fit(twos)
