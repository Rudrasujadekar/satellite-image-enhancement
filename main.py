import cv2

# Load original image
img = cv2.imread("images/satellite.jpg")

if img is None:
    print("Image not found!")

else:
    # Increase brightness and contrast
    enhanced = cv2.convertScaleAbs(
        img, alpha=1.15, beta=15
    )
    # SAVE ENHANCED IMAGE
    cv2.imwrite("images/enhanced.jpg",enhanced)

    # Histogram Equalization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    cv2.imwrite("images/histogram_equalized.jpg", equalized)

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite("images/gaussian_blur.jpg", gaussian)
    cv2.imshow("Gaussian Blur", gaussian)

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite("images/gaussian_blur.jpg", gaussian)
    cv2.imshow("Gaussian Blur", gaussian)

    # Median Blur
    median = cv2.medianBlur(img, 5)

    cv2.imwrite("images/median_blur.jpg", median)
    cv2.imshow("Median Blur", median)

    # Bilateral Filtering
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    cv2.imwrite("images/bilateral_filter.jpg", bilateral)
    cv2.imshow("Bilateral Filter", bilateral)

    # Image Sharpening
    import numpy as np

    kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
    ])

    sharpened = cv2.filter2D(img, -1, kernel)

    cv2.imwrite("images/sharpened.jpg", sharpened)
    cv2.imshow("Sharpened Image", sharpened)

    # Canny Edge Detection
    edges = cv2.Canny(img, 100, 200)

    cv2.imwrite("images/edge_detection.jpg", edges)
    cv2.imshow("Edge Detection", edges)

    # Show both images
    cv2.imshow("Original Image", img)
    cv2.imshow("Enhanced Image", enhanced)

    cv2.waitKey(0)
    cv2.destroyAllWindows()