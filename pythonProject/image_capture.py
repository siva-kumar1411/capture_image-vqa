import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Capture a single frame
ret, frame = cap.read()

if ret:
    # Display the captured frame
    cv2.imshow('Captured Image', frame)

    # Save the image
    cv2.imwrite('captured_image.jpg', frame)

    print("Image captured and saved as 'captured_image.jpg'")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not capture image")

cap.release()
# Load the captured image
image_path = 'captured_image.jpg'
image = cv2.imread(image_path)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Show the image with detected faces
cv2.imshow('Faces detected', image)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
