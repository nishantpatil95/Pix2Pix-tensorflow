# Pix2Pix-tensorflow

Tensorflow Link
https://www.tensorflow.org/tutorials/generative/pix2pix

![Alt Text](https://github.com/nishantpatil95/Pix2Pix-tensorflow/blob/master/Images/ToolUse1.gif)

![Alt text](https://github.com/nishantpatil95/Pix2Pix-tensorflow/blob/master/Images/Tool_Screenshot.PNG)

# How to use it

1. Download images which you want to train and store them in "./Database_Name/base".<br/>
In this case I have trained this model using with 40 modern house images.
Sample image : <br/>
![Alt Text](https://github.com/nishantpatil95/Pix2Pix-tensorflow/blob/master/Images/Original.jpg)<br>
2. run "image_sidebyside.py" which will save images side by side in "./Database_Name/train" folder.<br>
3.  open every images in "./Database_Name/train" folder in windows paint and apply color to each object e.g. window:red trees:green etc
you can add your own colors in paint tool<br/>

PaintToolObj=PaintTool()
PaintToolObj.AddButton("Walls","#FFFFFF")
PaintToolObj.AddButton("Window","#FF1C24")

![Alt Text](https://github.com/nishantpatil95/Pix2Pix-tensorflow/blob/master/Images/Imagesidebyside.png)

4 run pngtojpg to convert images in "./Database_Name/train" folder from .png to .jpg format.<br>
5 take some images and store them in ""./Database_Name/test" folder to track progress. <br>
6 run pixtopix.py(one iteration took 30 sec with GPU)<br>

![Alt Text](https://github.com/nishantpatil95/Pix2Pix-tensorflow/blob/master/Images/Training.gif)

5 each iterations result is stored in "./output/frame" folder.<br>
6 after 400-500 epoches open interactive tool and test your model. (To open tool set Open_Interactive_Tool=True in pixtopix.py)<br>
Input:<br>
![Alt Text](https://github.com/nishantpatil95/Pix2Pix-tensorflow/blob/master/Images/last%20(1).png)<br>
Result:<br>
![Alt Text](https://github.com/nishantpatil95/Pix2Pix-tensorflow/blob/master/Images/last%20(2).png)

