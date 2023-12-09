import cv2
import os
import time

cam = cv2.VideoCapture(0)

# cv2.namedWindow("Sample")

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
    path = 'D:\\Model\\face_recognition\\Dataset\\FaceData\\raw\\Tri'
    cv2.imwrite(os.path.join(path, img_name), frame)
    print("{} written!".format(img_name))
    img_counter += 1

    time.sleep(0.5)

cam.release()
cv2.destroyAllWindows()
