import os
import PIL
import cv2
import glob
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw, ImageGrab
from PIL import EpsImagePlugin
from keras.models import load_model
import io

# load the model
model = load_model('digitmodel.h5')

# specify ghostscript path
EpsImagePlugin.gs_windows_binary =  r'C:\Program Files\gs\gs9.55.0\bin\gswin64c'

# define functions
def clear_widget():
	global cv
	cv.delete("all")

def activate_event(event):
	global lastx, lasty
	cv.bind('<B1-Motion>', draw_lines)
	lastx, lasty = event.x, event.y

def draw_lines(event):
	global lastx, lasty
	x, y = event.x, event.y
	cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
	lastx, lasty = x, y

def recognize_digit():
	global image_number
	predictions = []
	percentage = []
	filename = f'image_{image_number}.png'
	widget=cv

	# save canvas image using ghostscript
	ps = cv.postscript(colormode='color')
	image = PIL.Image.open(io.BytesIO(ps.encode('utf-8')))
	image.save(filename)

	# read image
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	contours= cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 1)
		top = int(0.05*th.shape[0])
		bottom = top
		left = int(0.05*th.shape[1])
		right = left
		th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)

		roi= th[y-top:y+h+bottom, x-left:x+w+right]

		# resize to 28x28 pixels
		img = cv2.resize(roi, (28,28), interpolation=cv2.INTER_AREA)

		# reshape & normalize image to support model input
		img = img.reshape(1,28,28,1)
		img = img/255.0

		# predict the result
		pred = model.predict([img])[0]
		final_pred = np.argmax(pred)
		data = str(final_pred) + ' ' + str(int(max(pred)*100))+'%'

		font=cv2.FONT_HERSHEY_SIMPLEX
		fontScale=0.5
		color=(255,0,0)
		thickness=1

		cv2.putText(image, data, (x,y-5), font, fontScale, color, thickness)
		
	# show results in new window
	cv2.imshow('Digit', image)
	cv2.waitKey(0)

# create GUI
root = Tk()
root.resizable(0,0)
root.title("Digit Recognition App")

lastx, lasty = None, None
image_number = 0

# create canvas
cv = Canvas(root, width = 640, height = 480, bg='white', cursor='pencil')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

cv.bind('<Button-1>', activate_event)

# add buttons
btn_save = Button(root,text = 'Recognize',command = recognize_digit,width = 10,borderwidth=1,bg = '#242a44',fg = 'white',font = ('courier new',12))
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear =  Button(root,text = 'Clear',command = clear_widget,width = 10,borderwidth=1,bg = '#242a44',fg = 'white',font = ('courier new',12))
button_clear.grid(row=2, column=1, pady=1, padx=1)

root.mainloop()