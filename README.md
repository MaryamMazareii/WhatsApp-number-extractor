# WhatsApp Number Extractor

This project extracts WhatsApp-style phone numbers from video frames using OCR (Optical Character Recognition) and regex pattern matching.

---

## Features

- Extracts frames from a video at specified intervals.  
- Applies image preprocessing to improve OCR accuracy.  
- Uses Tesseract OCR to extract text from each frame.  
- Detects WhatsApp phone numbers formatted with country codes using regex.  
- Removes duplicates and saves extracted numbers to a CSV file.  
- Calculates accuracy metrics using sample (fake) phone numbers to respect privacy.

---

## Privacy Notice

- **No real personal data or videos are included in this repository.**  
- All phone numbers used for testing and accuracy calculation are **fake/sample numbers** only.  
- Video files containing actual personal data are excluded to comply with privacy policies.

---

## How It Works

1. Extracts frames from the input video at intervals (default 0.1 seconds → 10 FPS).  
2. Converts frames to grayscale, applies Gaussian blur, adaptive thresholding, and inverts colors to optimize OCR results.  
3. Runs Tesseract OCR on processed frames and searches for phone numbers matching WhatsApp number patterns using regex.  
4. Removes duplicate phone numbers and compiles the unique results into a CSV file.  
5. Compares extracted numbers against predefined fake ground truth numbers to compute accuracy metrics.

---

## Usage

1. Place your input video file in the project folder or update the `video_path` variable in the script.  
2. Install required dependencies:

   ```bash
   pip install opencv-python pytesseract pandas numpy
