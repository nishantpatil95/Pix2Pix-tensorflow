from glob import glob                                                           
import cv2 
pngs = glob('./Database/FrontView_Houses/train/*.png')

for j in pngs:
	print(j)
	img = cv2.imread(j)
	cv2.imwrite(j[:-3] + 'jpg', img)