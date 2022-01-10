import os
import numpy as np
import cv2 as cv


root_path = "antrenare\\"

names  = ["bart", "homer", "lisa", "marge"]

image_names = []
bboxes = []
characters = []
nb_examples = 0

for name in names:
	filename_annotations = root_path + name + ".txt"
	f = open(filename_annotations)
	for line in f:
		a = line.split(os.sep)[-1]
		b = a.split(" ")
		
		image_name = root_path + name + "\\" + b[0]
		bbox = [int(b[1]),int(b[2]),int(b[3]),int(b[4])]
		character = b[5][:-1]
		
		image_names.append(image_name)
		bboxes.append(bbox)
		characters.append(character)

width_hog = 36
height_hog = 36

low_yellow = (19, 90, 190)
high_yellow = (90, 255, 255)

for idx, img_name in enumerate(image_names):
	if idx < len(image_names)-1 and img_name == image_names[idx+1]:
		continue

	img = cv.imread(img_name)	
	num_rows = img.shape[0]
	num_cols = img.shape[1]

	# incerc sa generez maxim 30 de exemple negative care sa contina galben pentru fiecare poza, cu patch-uri de dimensiuni variabile de la 36x36, crescand cu puteri ale lui 1.5
	# astfel in datele negative imi salvez si patch-uri redimensionate deoarece fac sliding window de diferite dimensiuni
	for i in range(20):
		for j in range(4):
			x = np.random.randint(low=0, high=num_cols - (width_hog * (1.5**j)))
			y = np.random.randint(low=0, high=num_rows - (height_hog * (1.5**j)))

			in_bbox = 0
			aux_idx = idx
			while image_names[aux_idx] == img_name:
				if bboxes[aux_idx][0]-(18*(1.5**j)) <= x <= bboxes[aux_idx][2]+(18*(1.5**j)) and bboxes[aux_idx][1]-(18*(1.5**j)) <= y <= bboxes[aux_idx][3]+(18*(1.5**j)):
					in_bbox = 1
					break
				aux_idx -= 1
			if in_bbox == 1:
				continue	
			
			bbox_curent = [x, y, x+int((width_hog * (1.5**j))), y+int((height_hog * (1.5**j)))]
			xmin = bbox_curent[0]
			ymin = bbox_curent[1]
			xmax = bbox_curent[2]
			ymax = bbox_curent[3]
			negative_example = img[ymin:ymax, xmin:xmax]

			patch_hsv = cv.cvtColor(negative_example, cv.COLOR_BGR2HSV)
			yellow_patch = cv.inRange(patch_hsv, low_yellow, high_yellow)
			if yellow_patch.mean() < 50:
				continue

			neg_resized = cv.resize(negative_example, (height_hog, width_hog))
			filename = "antrenare\\data\\exempleNegative\\" + str(idx) + "_" + str(i)+str(j) + ".jpg"
			cv.imwrite(filename,neg_resized)		