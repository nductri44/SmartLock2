import time
import serial
import adafruit_fingerprint
import RPi.GPIO as GPIO
import imutils
from imutils.video import VideoStream
import cv2
import time
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("D:\\Model\\face_recognition\\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://facerecognition-49c2d-default-rtdb.asia-southeast1.firebasedatabase.app/'})

positive_ref = db.reference('/finger_positive')
negative_ref = db.reference('/finger_negative')

RELAY_PIN = 23

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

MAX_CONSECUTIVE_FAILURES = 5
DISABLE_DURATION = 30

consecutive_failures = 0
last_failure_time = 0

# If using with a computer such as Linux/RaspberryPi, Mac, Windows with USB/serial converter:
# uart = serial.Serial("/dev/ttyUSB0", baudrate=57600, timeout=1)

# If using with Linux/Raspberry Pi and hardware UART:
uart = serial.Serial("/dev/ttyS0", baudrate=57600, timeout=1)


finger = adafruit_fingerprint.Adafruit_Fingerprint(uart)


def get_fingerprint():
    print("Waiting for image...")
    while finger.get_image() != adafruit_fingerprint.OK:
        pass
    print("Templating...")
    if finger.image_2_tz(1) != adafruit_fingerprint.OK:
        return False
    print("Searching...")
    if finger.finger_search() != adafruit_fingerprint.OK:
        return False
    
    return True


# def get_fingerprint_detail():
#     print("Getting image...", end="")
#     i = finger.get_image()
#     if i == adafruit_fingerprint.OK:
#         print("Image taken")
#     else:
#         if i == adafruit_fingerprint.NOFINGER:
#             print("No finger detected")
#         elif i == adafruit_fingerprint.IMAGEFAIL:
#             print("Imaging error")
#         else:
#             print("Other error")
#         return False

#     print("Templating...", end="")
#     i = finger.image_2_tz(1)
#     if i == adafruit_fingerprint.OK:
#         print("Templated")
#     else:
#         if i == adafruit_fingerprint.IMAGEMESS:
#             print("Image too messy")
#         elif i == adafruit_fingerprint.FEATUREFAIL:
#             print("Could not identify features")
#         elif i == adafruit_fingerprint.INVALIDIMAGE:
#             print("Image invalid")
#         else:
#             print("Other error")
#         return False

#     print("Searching...", end="")
#     i = finger.finger_fast_search()

#     if i == adafruit_fingerprint.OK:
#         print("Found fingerprint!")
#         return True
#     else:
#         if i == adafruit_fingerprint.NOTFOUND:
#             print("No match found")
#         else:
#             print("Other error")
#         return False


def enroll_finger(location):
    for fingerimg in range(1, 3):
        if fingerimg == 1:
            print("Place finger on sensor...", end="")
        else:
            print("Place same finger again...", end="")

        while True:
            i = finger.get_image()
            if i == adafruit_fingerprint.OK:
                print("Image taken")
                break
            if i == adafruit_fingerprint.NOFINGER:
                print(".", end="")
            elif i == adafruit_fingerprint.IMAGEFAIL:
                print("Imaging error")
                return False
            else:
                print("Other error")
                return False

        print("Templating...", end="")
        i = finger.image_2_tz(fingerimg)
        if i == adafruit_fingerprint.OK:
            print("Templated")
        else:
            if i == adafruit_fingerprint.IMAGEMESS:
                print("Image too messy")
            elif i == adafruit_fingerprint.FEATUREFAIL:
                print("Could not identify features")
            elif i == adafruit_fingerprint.INVALIDIMAGE:
                print("Image invalid")
            else:
                print("Other error")
            return False

        if fingerimg == 1:
            print("Remove finger")
            time.sleep(1)
            while i != adafruit_fingerprint.NOFINGER:
                i = finger.get_image()

    print("Creating model...", end="")
    i = finger.create_model()
    if i == adafruit_fingerprint.OK:
        print("Created")
    else:
        if i == adafruit_fingerprint.ENROLLMISMATCH:
            print("Prints did not match")
        else:
            print("Other error")
        return False

    print("Storing model #%d..." % location, end="")
    i = finger.store_model(location)
    if i == adafruit_fingerprint.OK:
        print("Stored")
    else:
        if i == adafruit_fingerprint.BADLOCATION:
            print("Bad storage location")
        elif i == adafruit_fingerprint.FLASHERR:
            print("Flash storage error")
        else:
            print("Other error")
        return False

    return True


# def save_fingerprint_image(filename):
#     while finger.get_image():
#         pass

#     from PIL import Image

#     img = Image.new("L", (256, 288), "white")
#     pixeldata = img.load()
#     mask = 0b00001111
#     result = finger.get_fpdata(sensorbuffer="image")

#     x = 0
#     y = 0
#     for i in range(len(result)):
#         pixeldata[x, y] = (int(result[i]) >> 4) * 17
#         x += 1
#         pixeldata[x, y] = (int(result[i]) & mask) * 17
#         if x == 255:
#             x = 0
#             y += 1
#         else:
#             x += 1

#     if not img.save(filename):
#         return True
#     return False


def get_num(max_number):
    i = -1
    while (i > max_number - 1) or (i < 0):
        try:
            i = int(input("Enter ID # from 0-{}: ".format(max_number - 1)))
        except ValueError:
            pass
    return i


while True:
    print("----------------")
    if finger.read_templates() != adafruit_fingerprint.OK:
        raise RuntimeError("Failed to read templates")
    print("Fingerprint templates: ", finger.templates)
    if finger.count_templates() != adafruit_fingerprint.OK:
        raise RuntimeError("Failed to read templates")
    print("Number of templates found: ", finger.template_count)
    if finger.read_sysparam() != adafruit_fingerprint.OK:
        raise RuntimeError("Failed to get system parameters")
    print("Size of template library: ", finger.library_size)
    print("e) enroll print")
    print("f) find print")
    print("d) delete print")
    print("s) save fingerprint image")
    print("r) reset library")
    print("q) quit")
    print("----------------")
    c = input("> ")

    if c == "e":
        enroll_finger(get_num(finger.library_size))
    if c == "f":
        cap  = VideoStream(src=0).start()
        frame = cap.read()
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)
        if get_fingerprint():
            print("Detected #", finger.finger_id, "with confidence", finger.confidence)
            date_time = datetime.now().strftime("%d%m%Y%H%M%S")
            positive_ref.push({
                'Finger_ID': str(finger.finger_id),
                'Confidence': str(finger.confidence),
                'Detected_at': datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                'Datetime': date_time
            })
            image_path = 'D:\\Model\\flask_demo\\pythonlogin\\static\\finger_images\\positive\\{}.jpg'.format(date_time)
            cv2.imwrite(image_path, frame)
            print("Turning on...")
            GPIO.output(RELAY_PIN, 1)
            time.sleep(5)
            print("Turning off...")
            GPIO.output(RELAY_PIN, 0)
            consecutive_failures = 0
            last_failure_time = 0
        else:
            print("Finger not found")
            negative_ref.push({
                'Confidence': str(finger.confidence),
                'Detected_at': datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                'Datetime': date_time
            })
            image_path = 'D:\\Model\\flask_demo\\pythonlogin\\static\\finger_images\\positive\\{}.jpg'.format(date_time)
            cv2.imwrite(image_path, frame)
            consecutive_failures += 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print("Too many consecutive failures. Disabling for {} seconds.".format(DISABLE_DURATION))
            last_failure_time = time.time()
            consecutive_failures = 0
            time.sleep(DISABLE_DURATION)

    if c == "d":
        if finger.delete_model(get_num(finger.library_size)) == adafruit_fingerprint.OK:
            print("Deleted!")
        else:
            print("Failed to delete")
    # if c == "s":
    #     if save_fingerprint_image("fingerprint.png"):
    #         print("Fingerprint image saved")
    #     else:
    #         print("Failed to save fingerprint image")
    if c == "r":
        if finger.empty_library() == adafruit_fingerprint.OK:
            print("Library empty!")
        else:
            print("Failed to empty library")
    if c == "q":
        print("Exiting fingerprint example program")
        raise SystemExit
    