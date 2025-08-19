import cv2
import os
import math
import pytesseract
import re
import pandas as pd
import numpy as np

# ------------------ Video and image paths ------------------

video_path = '/content/phoneNumbers.mp4'   # Path to your video file
image_folder = 'frames'        # Folder to save frames
interval = 0.1                  # Interval in seconds (0.1 sec → 10 FPS)

# Create image folder if it doesn't exist
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# ------------------ Video → Frames -----------------

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():  # If the video was crashed
    print("Error: Cannot open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) # Manages the frame rate

frame_count = 0       # Frame counter
last_saved_time = -1  # Initialize


while True:
    # Ret is a boolean that is true since you start the video till it finishes
    # Frame is the real amount of frames of video
    ret, frame = cap.read()
    if not ret:
        break

    # Current time in seconds
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Round up to nearest 0.1 sec => we don't have data loss
    rounded_time = math.ceil(current_time / interval) * interval

   # Save frame only if this rounded time is new (preventing duplication)
    if rounded_time != last_saved_time:
        # Include both a sequential number and the timestamp in filename
        frame_filename = os.path.join(
            image_folder,
            f'frame_{frame_count:04d}_{rounded_time:.1f}.jpg'
        )
        cv2.imwrite(frame_filename, frame)
        last_saved_time = rounded_time
        frame_count+=1

cap.release()
print(f"Saved {frame_count} frames to '{image_folder}' folder.")  #Show what we stored (makes debugging easier)

# ------------------ OCR → DataFrame -----------------

# Initialize list to collect results
data = []

# Regex pattern for WhatsApp-style phone numbers: + followed by 8–15 digits
phone_pattern = r'\+\d{2}(?:\s?\d{2,4}){3,4}'

# Loop over all images (sorted by filename to keep order)
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        # Preprocessing for better OCR accuracy -> turning images into gray scale
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove noise with Gaussian blur
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Apply adaptive threshold (better than fixed 150 for different lighting)
        thresh = cv2.adaptiveThreshold(
          blur, 255,
          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv2.THRESH_BINARY,
          11, 2
        )

        # Invert if text is white on black
        thresh = cv2.bitwise_not(thresh)

        # Extract text with Tesseract
        text = pytesseract.image_to_string(thresh, lang="eng")

        # Split text in lines:
        lines = text.split("\n")

        for line in lines:
            # Find all phone numbers in the line
            phones_in_line = re.findall(phone_pattern, line)

            for phone in phones_in_line:
                # Normalize phone numbers
                clean_phone = phone.replace(" ", "")
                data.append({"Phone Number":clean_phone})

# Convert to DataFrame
df = pd.DataFrame(data)
df = df.drop_duplicates(subset="Phone Number",keep="first").reset_index(drop=True)

# Show DataFrame
print(df)

# Save to CSV
df.to_csv('whatsapp_numbers.csv', index=False)

#-----------------Accuracy check--------------------

# Suppose df contains extracted numbers
extracted_numbers = df["Phone Number"].values
# Ground truth numbers
ground_truth = np.array(["+12345789","+983728274516"])
# Check which extracted numbers are in ground truth
matches = np.isin(extracted_numbers, ground_truth)

# Accuracy = correct / total extracted
accuracy = matches.sum() / len(extracted_numbers) if len(extracted_numbers) > 0 else 0

print(f"Number of correctly extracted numbers: {matches.sum()}")
print(f"Total extracted numbers: {len(extracted_numbers)}")
print(f"Accuracy: {accuracy:.3f}")
recall_matches = np.isin(ground_truth, extracted_numbers)
recall = recall_matches.sum() / len(ground_truth)
print(f"Recall: {recall:.3f}")
errors = np.sum(ground_truth != extracted_numbers)
error_rate = errors / len(ground_truth)
print(f"Error Rate: {error_rate:.3f}")
mse = np.mean((ground_truth - extracted_numbers)**2)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")