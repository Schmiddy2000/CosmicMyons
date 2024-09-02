import cv2
import pytesseract
from PIL import Image
import os


def extract_frames(video_path, interval_seconds, output_folder):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_seconds)
    frame_count = 0
    saved_frame_count = 0
    success, frame = video.read()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_paths = []

    while success:
        if frame_count % interval_frames == 0:
            frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count}.png')
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
            saved_frame_count += 1

        success, frame = video.read()
        frame_count += 1

    video.release()
    return frame_paths


def crop_relevant_areas(image_path, counts_box, time_box):
    image = cv2.imread(image_path)
    counts_cropped = image[counts_box[1]:counts_box[3], counts_box[0]:counts_box[2]]
    time_cropped = image[time_box[1]:time_box[3], time_box[0]:time_box[2]]

    return counts_cropped, time_cropped


def ocr_extraction(cropped_image):
    text = pytesseract.image_to_string(Image.fromarray(cropped_image))
    return text.strip()


def process_video(video_path, counts_box, time_box, interval_seconds=1):
    # Step 1: Extract frames from the video
    frames = extract_frames(video_path, interval_seconds, 'extracted_frames')

    # Arrays to store the results
    coincidence_counts = []
    remaining_times = []

    # Step 2: Process each frame
    for frame_path in frames:
        print(frame_path)
        counts_cropped, time_cropped = crop_relevant_areas(frame_path, counts_box, time_box)

        # Step 3: Apply OCR to extract text
        counts_text = ocr_extraction(counts_cropped)
        time_text = ocr_extraction(time_cropped)

        # Step 4: Convert OCR results to numbers
        try:
            counts_value = int(counts_text)
        except ValueError:
            counts_value = None  # Handle OCR errors as None or a default value

        try:
            time_value = int(time_text)  # or float(time_text) if it's not an integer
        except ValueError:
            time_value = None  # Handle OCR errors as None or a default value

        # Store the values in the arrays
        if counts_value is not None:
            coincidence_counts.append(counts_value)
        if time_value is not None:
            remaining_times.append(time_value)

    return remaining_times, coincidence_counts


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

counts_box = (x1_counts, y1_counts, x2_counts, y2_counts)  # Coordinates for coincidence counts
time_box = (x1_time, y1_time, x2_time, y2_time)  # Coordinates for remaining time

remaining_times, coincidence_counts = process_video('/Users/lucas1/Downloads/IMG_8417 2_kurz.mp4', counts_box, time_box)
print(f'Remaining Times: {remaining_times}')
print(f'Coincidence Counts: {coincidence_counts}')
