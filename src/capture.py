import cv2
import os
import time

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

folder_name = input("Enter the name of the folder to save images: ")
folder_path = os.path.join('D:\\Model\\face_recognition\\Dataset\\FaceData\\raw', folder_name)

create_folder(folder_path)

cam = cv2.VideoCapture(0)

img_counter = 0
max_pictures = 100

# Load the cascade classifier
detector = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")

while img_counter < max_pictures:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
        cv2.putText(img, "Face Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
        cv2.putText(img, str(img_counter) + " images captured", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
        new_img = img[y:y+h, x:x+w]

        # Save the image only when a face is detected
        img_name = "{}.png".format(img_counter)
        cv2.imwrite(os.path.join(folder_path, img_name), new_img)
        print("{} written!".format(img_name))
        img_counter += 1

 
    cv2.imshow("image", img)

    time.sleep(0.5)

cv2.waitKey(0)
cv2.destroyAllWindows()
