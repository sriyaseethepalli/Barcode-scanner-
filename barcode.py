import cv2
import numpy as np
from pyzbar.pyzbar import decode

def detect_barcodes(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find barcodes in the image
    barcodes = decode(gray_image)

    # Loop over detected barcodes and draw rectangles around them
    for barcode in barcodes:
        # Extract barcode data and format it
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type

        # Get barcode bounding box points
        points = barcode.polygon

        # Convert the barcode points to a numpy array
        points = np.array(points, dtype=np.int32)

        # Draw a rectangle around the barcode
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

        # Put barcode data as text on the image
        text = f"{barcode_type}: {barcode_data}"
        cv2.putText(image, text, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Print barcode data to the console
        print(f"Detected Barcode Type: {barcode_type}")
        print(f"Detected Barcode Data: {barcode_data}")
        print("-" * 30)

    # Display the image with detected barcodes
    cv2.imshow('Barcode Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "F:\Competetions\BIRAC\Team Details\ID\Pavan ID CARD.jpeg"  # Replace with the actual path to your image
    detect_barcodes(image_path)
