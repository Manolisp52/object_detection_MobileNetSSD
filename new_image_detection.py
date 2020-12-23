#εισαγωγή απαραίτητων βιβλιοθηκών
import numpy as np
import argparse
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk,Image
import os

def obejct_det():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", default=img1)
	ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt")
	ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel")
	ap.add_argument("-c", "--confidence", type=float, default=0.2)
	args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	print("[INFO] loading model...")
	model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	# (note: normalization is done via the authors of the MobileNet SSD
	# implementation)
	image = cv2.imread(args["image"])
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	model.setInput(blob)
	detections = model.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# display the prediction
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("[INFO] {}".format(label))
			cv2.rectangle(image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# cv2.imshow("Output", image)
	# cv2.imwrite('image_detected.jpg',image)
	# cv2.waitKey(0)
	# img = Image.open(image)
	image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	width, height = image.size
	if int(width)<int(height): img = image.resize((400, 570), Image.ANTIALIAS)
	else: img = image.resize((900, 570), Image.ANTIALIAS)
	img = ImageTk.PhotoImage(img)
	panel.config(image=img)
	panel.image=img
    
def callback():
    global img1
    img1= filedialog.askopenfilename()
    obejct_det()
        


root=tk.Tk()
root.title("Object Detection-Ομάδα 12")
root.resizable(False, False)
window_height = 690
window_width = 1000
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate-35, y_cordinate-35))
root.configure(background='grey')
w1=tk.Label(root,text="Αναγνώριση Αντικειμένων",font="arial 40",bg="light blue")
w1.pack(fill="x")
panel = tk.Label(root,bg="grey")
panel.pack()
button=tk.Button(root,text="Επιλογή Εικόνας",font="arial 17",bg="grey",command=callback)
button.pack()
button.place(x=390,y=645)
root.mainloop()



