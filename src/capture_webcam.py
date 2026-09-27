"""
capture_webcam.py
------------------
Opens your Mac's webcam, shows a live preview, and lets you capture a photo
by pressing SPACE (saved to disk) -- useful for testing the trained model
on a real, live leaf photo instead of a dataset image.

Press SPACE to capture, ESC to quit without saving.

Usage:
    python capture_webcam.py --output ../data/test_images/my_leaf.jpg
"""

import argparse
import os

import cv2


def capture_photo(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: could not open webcam. Check camera permissions for Terminal "
              "in System Settings -> Privacy & Security -> Camera.")
        return

    print("Webcam open. Press SPACE to capture, ESC to cancel.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: failed to grab frame.")
            break

        cv2.imshow("Press SPACE to capture, ESC to cancel", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC
            print("Cancelled, no photo saved.")
            break
        elif key % 256 == 32:  # SPACE
            cv2.imwrite(output_path, frame)
            print(f"Saved photo to {output_path}")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a photo from the webcam.")
    parser.add_argument("--output", default="../data/test_images/webcam_capture.jpg",
                         help="Where to save the captured photo")
    args = parser.parse_args()

    capture_photo(args.output)
