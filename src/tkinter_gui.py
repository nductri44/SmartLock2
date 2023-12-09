import customtkinter
import subprocess

def face_recognition():
    print("Face Recognition Mode")
    subprocess.run(["python", "src\\face_rec_cam.py"])

def fingerprint_recognition():
    print("Finger Recognition Mode")
    subprocess.run(["python", "src\\finger_rec.py"])

app = customtkinter.CTk()
app.title("my app")
app.geometry("600x300")
app.grid_columnconfigure(0, weight=1)

button1 = customtkinter.CTkButton(app, text="Face ID", height=50, command=face_recognition)
button1.grid(row=3, column=0, pady=10, padx=10)

button2 = customtkinter.CTkButton(app, text="Finger ID", height=50, command=fingerprint_recognition)
button2.grid(row=4, column=0, pady=10, padx=10)

app.mainloop()
