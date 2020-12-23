# ΕΙΣΑΓΩΓΗ ΒΙΒΛΙΟΘΗΚΩΝ
import numpy as np
import argparse
import cv2
import tkinter as tk
from tkinter import filedialog


def search_image():            #ΕΠΙΣΤΡΟΦΗ  ΕΠΙΛΕΓΜΕΝΟΥ ΑΡΧΕΙΟΥ ΜΕ ΤΗΝ ΕΝΤΟΛΗ filedialog
    global image1
    image1 = filedialog.askopenfilename()
    root.destroy()

    return image1

root = tk.Tk()              #ΔΗΜΙΟΥΡΓΙΑ GUI
root.geometry('800x800')
root.resizable(False, False)
root.title('Oμαδα-12')
w = tk.Label(root, text = "IMAGE-DETECTION-12", font = "Arial 36", bg ='lightgray', width = 900)
w.pack()
button = tk.Button(root, text = "EΠΙΛΟΓΗ", font = "Arial 36", command = search_image)
button.pack()

root.mainloop()


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=image1,
	help="path to input image")
ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


print("[INFO] loading model...")
model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)


print("[INFO] computing object detections...")
model.setInput(blob)
detections = model.forward()

for i in np.arange(0, detections.shape[2]):

	confidence = detections[0, 0, i, 2]

	if confidence > args["confidence"]:
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

	
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))
		cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(image, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


cv2.imshow("Output", image)
cv2.imwrite('image_detected.jpg',image)
cv2.waitKey(0)
