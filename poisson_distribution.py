# Imports
import cv2
import os
import pytesseract
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json


def save_to_json(file_path, remaining_times, coincidence_counts):
    data = {
        'remaining_times': remaining_times,
        'coincidence_counts': coincidence_counts
    }

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return None


def read_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    remaining_times = data.get('remaining_times', [])
    coincidence_counts = data.get('coincidence_counts', [])

    return remaining_times, coincidence_counts


# Get the number of images to calculate the range for the loop over all images
def count_files_in_directory(directory_path):
    path = Path(directory_path)
    return len([f for f in path.iterdir() if f.is_file()])


# Extract images from the video in 1-second intervals
def extract_frames(video_path, interval_seconds, output_folder) -> None:
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_seconds)
    frame_count = 0
    saved_frame_count = 0
    success, frame = video.read()

    print(fps)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while success:
        if frame_count % interval_frames == 0:
            frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count}.png')
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        success, frame = video.read()
        frame_count += 1

    video.release()

    return None


v_path = "/Users/lucas1/Downloads/IMG_8417 Kopie_kurz.mp4"
v_path_2 = "/Users/lucas1/Downloads/IMG_8417 2_kurz.mp4"
v_path_3 = "/Users/lucas1/Downloads/IMG_8417_first_part.mp4"
v_path_4 = "/Users/lucas1/Downloads/IMG_8417_first_and_second_start.mp4"
final_v_path = "/Users/lucas1/Downloads/IMG_8417 2.mp4"

# Beispielnutzung
extract_frames(final_v_path, 1, 'extracted_frames')


# Preprocess the images to reduce noise and make the shapes more apparent.
# This greatly improves the performance of the OCR program
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
    denoised_image = cv2.fastNlMeansDenoising(threshold_image, None, 30, 7, 21)
    return denoised_image


# Create cropped images for the regions in interest
def crop_relevant_areas(image_path, counts_box, time_box, output_folder):
    image = preprocess_image(image_path)
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    counts_cropped = image[counts_box[1]:counts_box[3], counts_box[0]:counts_box[2]]
    time_cropped = image[time_box[1]:time_box[3], time_box[0]:time_box[2]]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counts_output_path = os.path.join(output_folder, f'counts_{os.path.basename(image_path)}')
    time_output_path = os.path.join(output_folder, f'time_{os.path.basename(image_path)}')

    cv2.imwrite(counts_output_path, counts_cropped)
    cv2.imwrite(time_output_path, time_cropped)


# Example usage:
x_max = 720 - 1
y_max = 616 - 1

x1_counts = 100
x2_counts = x_max
y1_counts = 50
y2_counts = 300

x1_time = 0
x2_time = 200
y1_time = y_max - 75
y2_time = y_max - 25

counts_box = (x1_counts, y1_counts, x2_counts, y2_counts)  # Define these based on your needs
time_box = (x1_time, y1_time, x2_time, y2_time)  # Define these based on your needs
test_image_path = "extracted_frames/frame_0.png"

# crop_relevant_areas(test_image_path, counts_box, time_box, 'cropped_frames')


# Extract the text from the images with the additional information to only expect digits
def ocr_extraction(cropped_image_path):
    # text = pytesseract.image_to_string(Image.open(cropped_image_path))
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(Image.open(cropped_image_path), config=custom_config).strip()
    return text


# Loop over multiple images
do_loop = True

counts = []
times = []
error_indices = []
if do_loop:
    for i in tqdm(range(count_files_in_directory("extracted_frames"))):
        image_path = f"extracted_frames/frame_{i}.png"
        crop_relevant_areas(image_path, counts_box, time_box, 'cropped_frames')

        # Example usage:
        counts_text = ocr_extraction(f'cropped_frames/counts_frame_{i}.png')
        time_text = ocr_extraction(f'cropped_frames/time_frame_{i}.png')

        # Step 4: Convert OCR results to numbers
        try:
            counts_value = int(counts_text)
        except ValueError:
            counts_value = None  # Handle OCR errors as None or a default value

        try:
            time_value = int(float(time_text))  # or float(time_text) if it's not an integer
        except ValueError:
            time_value = None  # Handle OCR errors as None or a default value

        if counts_value is not None and time_value is not None:
            counts.append(counts_value)
            times.append(time_value)
        else:
            error_indices.append(i)

        # Store the values in the arrays
        # if counts_value is not None:
        #     coincidence_counts.append(counts_value)
        # if time_value is not None:
        #     remaining_times.append(time_value)

        # counts.append(int(counts_text))
        # times.append(int(float(time_text)))

        # print(f'Index: {i}, Counts: {counts_text}, Time Remaining: {time_text}')

    print(counts)
    print(times)
    print(error_indices)


# Write a function that can
# - handle the issues that arise in the time-text recognition ('.')
# - Distinguish the 1-second and 2-second measurements
# - Compute the counts (already for a histogram?)
# - function that can save the data as JSON or load it

test_times = [41, 41, 390, 39, 370, 350, 35, 34, 34, 320, 32, 30, 30, 28, 26, 26, 24, 24, 23, 23, 210, 21, 19, 170,
              170, 15, 15, 14, 14, 12, 120, 100, 100, 8, 60, 6, 5, 5, 3, 3, 10]

test_times_2 = [41, 41, 390, 39, 370, 350, 35, 34, 34, 320, 32, 30, 30, 28, 26, 26, 24, 24, 23, 23, 210, 21, 19, 170,
                170, 15, 15, 14, 14, 12, 120, 100, 100, 8, 60, 6, 5, 5, 3, 3, 10, 1, 599, 5990, 599, 599, 595, 595,
                593, 591, 591, 589, 589, 5880]


def clean_times_array(times_array, counts_array):
    entries = len(times_array)
    mean_range = 5
    current_index = 0
    not_one_anymore = False
    reset_counter = 0

    print(type(times_array[0]), type(counts_array[0]))

    cleaned_counts_array = counts_array[mean_range:-mean_range]
    cleaned_times_array = []

    # Could 'i' replace current_index here?
    for i in range(entries - 2 * mean_range):
        pre_mean = np.mean(times_array[current_index:mean_range + current_index])
        post_mean = np.mean(times_array[current_index + mean_range + 1:current_index + 2 * mean_range + 1])
        current_value = times_array[mean_range + i]

        current_index += 1

        # Also do a check that maybe uses the post-mean to see if the time index has been restarted (new measurement)
        # Use post-mean for 5 iterations
        if not_one_anymore and reset_counter < 5:
            print('Not one anymore:', current_value, post_mean + 10)
            if current_value > post_mean + 10:
                current_value = round(current_value / 10)
                times_array[mean_range + i] = current_value

            # Reset the reset counter and the was_one flag
            if reset_counter == 4:
                not_one_anymore = False
                reset_counter = 0
            pass
        else:
            # Use pre-mean
            if pre_mean < current_value:
                current_value = round(current_value / 10)
                times_array[mean_range + i] = current_value

            if current_value == 1 and times_array[mean_range + i + 1] > 10:
                not_one_anymore = True

        cleaned_times_array.append(current_value)

        print(f'index: {i}, pre_mean: {pre_mean}, current_value: {current_value}, post_mean: {post_mean}')
        print(len(cleaned_times_array), len(cleaned_counts_array))

    return cleaned_times_array, cleaned_counts_array


extracted_times, extracted_counts = clean_times_array(times, counts)

save_to_json('test_data_1.json', extracted_times, extracted_counts)

my_times, my_counts = read_from_json('test_data_1.json')

print(my_times)
print(my_counts)
