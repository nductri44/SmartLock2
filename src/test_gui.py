import pickle
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, PhotoImage
from PIL import Image, ImageTk
import cv2
import os
import imutils
import pytz

from imutils.video import VideoStream
import align.detect_face
import facenet
import tensorflow as tf
import numpy as np


names = set()
utc = pytz.UTC


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global names
        global open_webcam
        open_webcam = 'false'
        with open("src/nameslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                names.add(i)
        self.title_font = tkfont.Font(
            family='Helvetica', size=16, weight="bold")
        self.title("Face Recognizer")
        self.resizable(False, False)
        app_width = 1200
        app_height = 600
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = (screen_width / 2) - (app_width / 2)
        y = (screen_height / 2) - (app_height / 2)

        self.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        container = tk.Frame(self)
        container.place(relx=0.5, rely=0.5, anchor='center')
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageFour, PageTakeFace, PageDetectFace):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            self.destroy()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        load1 = Image.open("src/Touch_ID.png")
        load1 = load1.resize((250, 250), Image.ANTIALIAS)
        render1 = PhotoImage(file='src/Touch_ID.png')

        render1 = ImageTk.PhotoImage(Image.open(
            "src/Touch_ID.png").resize((250, 250), Image.ANTIALIAS))

        button4 = tk.Button(
            self, image=render1, command=lambda: self.controller.show_frame("PageTwo"))
        button4.image = render1
        button4.grid(row=0, column=0, rowspan=4,
                     padx=10, pady=12, sticky="nsew")

        load2 = Image.open("src/face-id-id.png")
        load2 = load2.resize((50, 50), Image.ANTIALIAS)
        render2 = PhotoImage(file='src/face-id-id.png')

        render2 = ImageTk.PhotoImage(Image.open(
            "src/face-id-id.png").resize((250, 250), Image.ANTIALIAS))

        button5 = tk.Button(
            self, image=render2, command=lambda: self.controller.show_frame("PageOne"))
        button5.image = render2
        button5.grid(row=1, column=1, rowspan=4,
                     padx=10, pady=12, sticky="nsew")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            self.controller.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        color1 = '#020f12'
        color2 = '#05d7ff'
        color3 = '#65e7ff'
        color4 = 'BLACK'
        color5 = 'YELLOW'
        self.buttoncanc = tk.Button(self,
                                    background=color2,
                                    foreground=color4,
                                    activebackground=color3,
                                    activeforeground=color4,
                                    highlightthickness=2,
                                    highlightbackground=color2,
                                    width=15,
                                    height=2,
                                    border=0,
                                    cursor='hand2',
                                    text="Cancel",
                                    font=('Arial', 16, 'bold'),
                                    command=lambda: controller.show_frame("StartPage"))
        self.buttoncanc.place(relx=0.5, rely=0.5, anchor='center')

        self.buttonTakeFace = tk.Button(self,
                                        background=color2,
                                        foreground=color4,
                                        activebackground=color3,
                                        activeforeground=color4,
                                        highlightthickness=2,
                                        highlightbackground=color2,
                                        width=15,
                                        height=2,
                                        border=0,
                                        cursor='hand2',
                                        text="Add new",
                                        font=('Arial', 16, 'bold'), command=lambda: controller.show_frame("PageTakeFace"))
        self.buttonTakeFace.place(relx=0.5, rely=0.3, anchor='center')

        self.buttonTakeFace = tk.Button(self,
                                        background=color2,
                                        foreground=color4,
                                        activebackground=color3,
                                        activeforeground=color4,
                                        highlightthickness=2,
                                        highlightbackground=color2,
                                        width=15,
                                        height=2,
                                        border=0,
                                        cursor='hand2',
                                        text="Detect",
                                        font=('Arial', 16, 'bold'), command=lambda: controller.show_frame("PageDetectFace"))
        self.buttonTakeFace.place(relx=0.5, rely=0.1, anchor='center')


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        color1 = '#020f12'
        color2 = '#05d7ff'
        color3 = '#65e7ff'
        color4 = 'BLACK'
        color5 = 'YELLOW'

        self.buttoncanc = tk.Button(self,
                                    background=color2,
                                    foreground=color4,
                                    activebackground=color3,
                                    activeforeground=color4,
                                    highlightthickness=2,
                                    highlightbackground=color2,
                                    width=15,
                                    height=2,
                                    border=0,
                                    cursor='hand2',
                                    text="Cancel",
                                    font=('Arial', 16, 'bold'),
                                    command=lambda: controller.show_frame("StartPage"))
        self.buttoncanc.place(relx=0.5, rely=0.5, anchor='center')

        self.buttonTakeFace = tk.Button(self,
                                        background=color2,
                                        foreground=color4,
                                        activebackground=color3,
                                        activeforeground=color4,
                                        highlightthickness=2,
                                        highlightbackground=color2,
                                        width=15,
                                        height=2,
                                        border=0,
                                        cursor='hand2',
                                        text="New Fingerprint",
                                        font=('Arial', 16, 'bold'), command=lambda: controller.show_frame("PageThree"))
        self.buttonTakeFace.place(relx=0.5, rely=0.3, anchor='center')

        self.buttonTakeFace = tk.Button(self,
                                        background=color2,
                                        foreground=color4,
                                        activebackground=color3,
                                        activeforeground=color4,
                                        highlightthickness=2,
                                        highlightbackground=color2,
                                        width=15,
                                        height=2,
                                        border=0,
                                        cursor='hand2',
                                        text="Detect Finger",
                                        font=('Arial', 16, 'bold'), command=lambda: controller.show_frame("PageDetectFingert"))
        self.buttonTakeFace.place(relx=0.5, rely=0.1, anchor='center')



class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller


### PAGE TAKE FACES ###
num_of_images = 0


class PageTakeFace(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        global cam_on
        cam_on = False
        global cap
        cap = None

        def display_frame():
            global cam_on
            global num_of_images
            detector = cv2.CascadeClassifier(
                "src/haarcascade_frontalface_default.xml")

            id = stundent_id_entry.get()

            filepath = 'src/raw/' + id

            isExist = os.path.exists(filepath)

            if not isExist:
                print('The new directory is created!')
                print(filepath)
                os.makedirs(filepath)

            if cam_on:

                ret, frame = cap.read()

                if ret:
                    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    filename = '.'.join([str(num_of_images), 'jpg'])
                    path = os.path.join(filepath, filename)
                    cv2.imwrite(path, frame)
                    face = detector.detectMultiScale(
                        image=opencv_image, scaleFactor=1.1, minNeighbors=5)
                    for x, y, w, h in face:
                        cv2.rectangle(frame, (x, y),
                                      (x+w, y+h), (8, 238, 255), 2)
                        cv2.putText(frame, "Face Detected", (x, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (8, 238, 255))
                        cv2.putText(frame, str(str(num_of_images)+" images captured"),
                                    (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (8, 238, 255))
                        # new_img = frame[y:y+h, x:x+w]

                    # Capture the latest frame and transform to image
                    captured_image = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))

                    # Convert captured image to photoimage
                    photo_image = ImageTk.PhotoImage(
                        captured_image.resize((500, 300), Image.ANTIALIAS))

                    # Displaying photoimage in the label
                    label_widget.photo_image = photo_image

                    # Configure image in the label
                    label_widget.configure(image=photo_image)

                    # Repeat the same process after every 10 seconds
                    num_of_images += 1

                    if num_of_images == 21:
                        stop_vid()
                        num_of_images = 0
                        messagebox.showinfo(
                            "INSTRUCTIONS", "We captured 20 pic of your Face.")
                        return 'ok'

                label_widget.after(10, display_frame)

        def start_vid():
            global cam_on, cap
            stop_vid()
            cam_on = True
            cap = cv2.VideoCapture(1)
            display_frame()

        def stop_vid():
            label_widget.configure(image=None)
            label_widget.configure(image="")

            global cam_on
            cam_on = False

            if cap:
                cap.release()

        user_info_frame = tk.LabelFrame(self, text="User Information")
        user_info_frame.grid(row=0, column=0, padx=20, pady=10)

        first_name_label = tk.Label(user_info_frame, text="Your Name")
        first_name_label.grid(row=0, column=0)
        student_id_label = tk.Label(user_info_frame, text="Your Student Id")
        student_id_label.grid(row=0, column=1)

        first_name_entry = tk.Entry(user_info_frame)
        stundent_id_entry = tk.Entry(user_info_frame)
        first_name_entry.grid(row=1, column=0)
        stundent_id_entry.grid(row=1, column=1)

        buttoncanc = tk.Button(user_info_frame, text="Cancel", bg="#ffffff",
                               fg="#263942", command=lambda: controller.show_frame("StartPage"))
        buttoncanc.grid(row=2, column=0, pady=10, ipadx=5, ipady=4)

        buttoncanc = tk.Button(user_info_frame, text="Confirm", bg="#ffffff",
                               fg="#263942", command=start_vid)
        buttoncanc.grid(row=2, column=1, pady=10, ipadx=5, ipady=4)

        label_widget = tk.Label(self)
        label_widget.grid(row=3, column=0)

        for widget in user_info_frame.winfo_children():
            widget.grid_configure(padx=10, pady=5)


# Face Recognition part

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")
with tf.Graph().as_default():

    # Cai dat GPU neu co
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():

        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(FACENET_MODEL_PATH)

        # Get input and output tensors
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")


### PAGE DETECT FACES ###
class PageDetectFace(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        global cam_detect_on
        cam_detect_on = False
        global cap_detect
        cap_detect = None
        global count_unknown
        count_unknown = 0
        global detect_time
        detect_time = 0
        global mode
        mode = 1


        def detect_frame():
            global cam_detect_on
            global count_unknown
            global detect_time
            if cam_detect_on:
                ret, frame = cap_detect.read()

                if ret:
                    frame = imutils.resize(frame, width=600)
                    frame = cv2.flip(frame, 1)
                    bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                    faces_found = bounding_boxes.shape[0]

                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=(np.int32))
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]
                        print(bb[i][3] - bb[i][1])
                        print(frame.shape[0])
                        print((bb[i][3] - bb[i][1]) / frame.shape[0])
                        if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                            cropped = frame[bb[i][1]:bb[i]
                                            [3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(
                                cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=(cv2.INTER_CUBIC))
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(
                                -1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {
                                images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(
                                embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(
                                predictions, axis=1)
                            best_class_probabilities = predictions[(
                                np.arange(len(best_class_indices)), best_class_indices)]
                            best_name = class_names[best_class_indices[0]]

                            count_unknown += 1
                            if best_class_probabilities > 0.8:
                                print('Name: {}, Probability: {}'.format(
                                    best_name, best_class_probabilities))
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0,
                                                                                                  255,
                                                                                                  0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                cv2.putText(frame, best_name, (text_x, text_y), (cv2.FONT_HERSHEY_COMPLEX_SMALL), 1,
                                            (255, 255, 255), thickness=1, lineType=2)
                                cv2.putText(frame, (str(round(best_class_probabilities[0], 3))), (text_x, text_y + 17), (cv2.FONT_HERSHEY_COMPLEX_SMALL),
                                            1, (255, 255, 255), thickness=1, lineType=2)
                                detect_time += 1
                            else:
                                print('Unknown')


        def start_check_in():
            global cam_detect_on, cap_detect, mode
            stop_detect()
            cam_detect_on = True
            cap_detect = cv2.VideoCapture(1)
            mode = 1
            detect_frame()

        def stop_detect():
            detect_widget.configure(image=None)
            detect_widget.configure(image="")

            global cam_detect_on
            cam_detect_on = False

            if cap_detect:
                cap_detect.release()

        ####### detect face screen #######

        buttoncanc1 = tk.Button(self, text="Cancel", bg="#ffffff",
                                fg="#263942", command=lambda: controller.show_frame("StartPage"))
        buttoncanc1.place(relx=0.5, rely=0.5, anchor='center')

        buttoncanc3 = tk.Button(self, text="Stop Detect", bg="#ffffff",
                                fg="#263942", command=stop_detect)
        buttoncanc3.place(relx=0.5, rely=0.3, anchor='center')

        buttoncanc2 = tk.Button(self, text="Face Check In", bg="#ffffff",
                                fg="#263942", command=start_check_in)
        buttoncanc2.place(relx=0.5, rely=0.1, anchor='center')

        detect_widget = tk.Label(self)
        detect_widget.place(relx=0.5, rely=0.7, anchor='center')


app = MainUI()
app.iconphoto(False, tk.PhotoImage(file='src/icon.ico'))
app.mainloop()