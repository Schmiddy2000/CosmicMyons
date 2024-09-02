# Imports
import cv2
import os
import pytesseract
from PIL import Image

# Set the tesseract_cmd to the installed path if it's not being detected automatically
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'


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
# Beispielnutzung
extract_frames(v_path_2, 1, 'extracted_frames')


def crop_relevant_areas(image_path, counts_box, time_box, output_folder):
    image = cv2.imread(image_path)
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
test_image_path = "/Users/lucas1/Downloads/frame_0.png"

crop_relevant_areas(test_image_path, counts_box, time_box, 'cropped_frames')


def ocr_extraction(cropped_image_path):
    text = pytesseract.image_to_string(Image.open(cropped_image_path))
    return text


# Example usage:
counts_text = ocr_extraction('/Users/lucas1/Downloads/counts_frame_0.png')
time_text = ocr_extraction('/Users/lucas1/Downloads/time_frame_0.png')

print(f'Counts: {counts_text}')
print(f'Time Remaining: {time_text}')
