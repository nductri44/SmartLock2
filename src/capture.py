import cv2
import os
import time

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

folder_name = input("Enter the name of the folder to save images: ")
folder_path = os.path.join(r'D:\\Model\\face_recognition\\Dataset\\FaceData\\raw', folder_name)

create_folder(folder_path)

cam = cv2.VideoCapture(0)

img_counter = 0
max_pictures = 50

while img_counter < max_pictures:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("sample", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break

    img_name = "{}.png".format(img_counter)
    cv2.imwrite(os.path.join(folder_path, img_name), frame)
    print("{} written!".format(img_name))
    img_counter += 1

    time.sleep(0.5)

cam.release()
cv2.destroyAllWindows()
