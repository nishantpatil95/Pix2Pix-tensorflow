import tensorflow as tf

import os
import time
import sys
from matplotlib import pyplot as plt
from IPython import display
import datetime
from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageTk,Image  ,ImageDraw
import PIL
import collections


BUFFER_SIZE = 50
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512
OUTPUT_CHANNELS = 3
LAMBDA = 100
CURRENT_EPOCH=0;    #Set this if training stops in between.
RESTORE_CKP=True   #True For starting from checkpoint and For testing in interactive tool
DATBASE_PATH='./Database/HousesWithVector' #give database path here
OUTPUT_DIR="./output"
Open_Interactive_Tool=True  #set True to open interactive tool


CKP_SAVE_INT=10 #chkpt interval
EPOCHS = 1000 #iterations

def MKDIR(Dir):
	if not os.path.isdir(Dir):
		os.mkdir(Dir)

MKDIR(OUTPUT_DIR)
MKDIR("./Test")
MKDIR(OUTPUT_DIR+"/frame")
############################################# PAINT STUFF ############################################

#os.system('xset r off')
class PaintTool(object):
	CurrentColor='#000000'
	BrushSize=5
	ListOfButtons=[]
	FillColor=False
	Dir=0
	Snap=False
	Xold=0
	Yold=0
	moves=0
	capture=False;
	def keyup(self,e):
		#print('up', e.char)
		self.Snap=False
	def keydown(self,e):
		self.Snap=True
		#print('down', e.char)
		
		
	def	round(self, n):
		n=n-n%(self.BrushSize*self.choose_size_button.get()*2)
		return n

	def	SetDir(self,x):
		self.Dir=x
		self.capture=True
	
	def __init__(self):
		self.root = Tk()
		
		color_button = Button(self.root, text='color', command=self.choose_color)
		color_button.grid(row=1, column=0)
		
		
		
		self.choose_size_button = Scale(self.root, from_=1, to=10, orient=VERTICAL)
		self.choose_size_button.grid(row=2, column=0)
		
		fill_button = Button(self.root, text='Fill', command=self.fill_color_change_mode)
		fill_button.grid(row=3, column=0)
		
		fill_button = Button(self.root, text='Ver', command=lambda:self.SetDir(1))
		fill_button.grid(row=4, column=0)
		
		fill_button = Button(self.root, text='Hor', command=lambda:self.SetDir(2))
		fill_button.grid(row=5, column=0)
		
		fill_button = Button(self.root, text='Free', command=lambda:self.SetDir(0))
		fill_button.grid(row=6, column=0)
		
		self.c = Canvas(self.root, bg='black', width=512, height=512)
		self.c.grid(row=1, column=1 ,rowspan=5,columnspan=5)
		self.c.bind('<B1-Motion>', self.MouseDown)
		self.c.bind('<Button-1>', self.MouseClicked)
		self.c.bind("<KeyPress>", self.keydown)
		self.c.bind("<KeyRelease>", self.keyup)
		self.c.focus_set()
		self.c.bind('<ButtonRelease>', self.MouseRelease)
		
		self.go_btn = Button(self.root, text='>', command=self.convert)
		self.go_btn.grid(row=1, column=6 ,rowspan=5)
		
		image=Image.open("./Test/Result.png")
		image=image.resize([512, 512])
		img=ImageTk.PhotoImage(image)
		self.resultlable=Label(self.root,image=img)
		self.resultlable.image=img
		self.resultlable.grid(row=1, column=7 ,rowspan=5,columnspan=5)
		self.Imagetest=Image.new("RGB",(512,512))
		self.DrawtestImage=ImageDraw.Draw(self.Imagetest)
		self.UpdateLeftPanel()
		self.UpdateRightPanel()
	
	def convert(self):
		self.UpdateRightPanel()
	
	def fill_color_change_mode(self):
		self.FillColor=True
	
	
	def MouseClicked(self,event):
		X=event.x
		Y=event.y
		if self.FillColor:
			self.FillColor=False
			ReplaceColor=self.Imagetest.getpixel((X,Y))
			if ReplaceColor!=self.CurrentColor:
				self.FloodFillColor(X,Y,ReplaceColor,1)
			return
			
	def FloodFillColor(self,X,Y,ReplaceColor,Limit):
		#print(X,Y)
		
		
		for i in range(512):
			print("fill ",str(i))
			for j in range(512):
				source=self.Imagetest.getpixel((i,j))
				if source == ReplaceColor:
					self.DrawtestImage.point((i, j), fill=self.CurrentColor)
					self.c.create_line(i, j, i+1, j, fill=self.CurrentColor)
		
		
		#Flood fill lags
		"""self.DrawtestImage.floodfill((X, Y),fill=self.CurrentColor)
		self.c.floodfill((X, Y),fill=self.CurrentColor)
		source=self.Imagetest.getpixel((X,Y))
		if source == ReplaceColor:
			#self.DrawtestImage.rectangle((X, Y, X, Y), fill=self.CurrentColor, outline=self.CurrentColor,width=1)
			self.c.create_rectangle(X, Y, X, Y, fill=self.CurrentColor,outline=self.CurrentColor)
			if(Limit==100):
				return
			if X>=1 and X<511 and Y>1 and Y<511:
				self.FloodFillColor(X+1,Y,ReplaceColor,Limit+1)
			if X>=1 and X<511 and Y>1 and Y<511:
				self.FloodFillColor(X+1,Y+1,ReplaceColor,Limit+1)
			if X>=1 and X<511 and Y>1 and Y<511:
				self.FloodFillColor(X,Y+1,ReplaceColor,Limit+1)
			if X>=1 and X<511 and Y>1 and Y<511:
				self.FloodFillColor(X-1,Y+1,ReplaceColor,Limit+1)"""

		

    
		

	
	def MouseDown(self,event):
		#print(event.x,event.y)
		size=self.choose_size_button.get()
		X=event.x
		Y=event.y
		if self.capture:
			self.capture=False
			self.Xold=X
			self.Yold=Y
		
			
		if self.Snap:
			X=self.round(event.x)
			Y=self.round(event.y)
			
			
			
		if self.Dir==1:
			X=self.Xold
		elif self.Dir==2:
			Y=self.Yold
		
			
		x0=X-self.BrushSize*size
		y0=Y-self.BrushSize*size
		x1=X+self.BrushSize*size
		y1=Y+self.BrushSize*size
		
			
		self.c.create_rectangle(x0, y0, x1, y1, fill=self.CurrentColor,outline=self.CurrentColor)
		#self.DrawtestImage.line([x0, y0, x1, y1],fill=self.CurrentColor, width=self.BrushSize*size)
		self.DrawtestImage.rectangle((x0, y0, x1, y1), fill=self.CurrentColor, outline=self.CurrentColor,width=self.BrushSize*size)
		if self.Dir==1:
			self.Yold=event.y
		elif self.Dir==2:
			self.Xold=event.x
		
		
	
	def CustomColor(self,Color):
		self.CurrentColor=Color
		
	
	def AddButton(self,Name,Color):
		btn=Button(self.root, text=Name, command=lambda: self.CustomColor(Color))
		btn.grid(row=0, column=len(self.ListOfButtons)+1)
		self.ListOfButtons.append({'name':Name,'color':Color})
	
	def UpdateLeftPanel(self):
		self.Imagetest.save("./Test/TestImage.png")
		self.Imagetest.save("./Test/Input_"+str(self.moves+1)+".png")
		imagesidebyside=Image.new('RGB', (self.Imagetest.width + self.Imagetest.width, self.Imagetest.height))
		imagesidebyside.paste(self.Imagetest, (0, 0))
		imagesidebyside.paste(self.Imagetest, (self.Imagetest.width, 0))
		imagesidebyside.save("./Test/TestImageSideBySide.png")
		
	def MouseRelease(self,event):
		self.UpdateLeftPanel()
		self.UpdateRightPanel()
		if not self.Dir==0:
			self.capture=True
		
	def UpdateRightPanel(self):
		custom_test_dataset = tf.data.Dataset.list_files('./Test/TestImageSideBySide*')
		custom_test_dataset = custom_test_dataset.map(load_image_test)
		custom_test_dataset = custom_test_dataset.batch(BATCH_SIZE)
		#print("here")
		self.moves=self.moves+1
		for example_input, example_target in custom_test_dataset.take(1):
			generate_result_image(generator, example_input, example_target,self.moves)
		#print("here")
		image=Image.open("./Test/Result.png")
		image=image.resize([512, 512])
		img = ImageTk.PhotoImage(image)   
		self.resultlable.configure(image=img)
		self.resultlable.image=img
		#print("here")
		
	def SetGenerator(self,Model):
		self.model=Model
		
	
	def choose_color(self):
		self.CurrentColor = askcolor(color=self.CurrentColor)[1]
	
	def Run(self):
		while True:
			self.root.mainloop()
		print("WHYYY")
		

############################################# PAINT STUFF ############################################	


"""PaintToolObj=PaintTool()
PaintToolObj.AddButton("Walls","#FFFFFF")
PaintToolObj.AddButton("Window","#FF1C24")
PaintToolObj.AddButton("Door","#880015")
PaintToolObj.AddButton("Pool","#3F48CC")
PaintToolObj.AddButton("Tree","#22B14C")
PaintToolObj.AddButton("Sky","#00A2E8")
PaintToolObj.AddButton("Stairs","#FF7F27")
PaintToolObj.AddButton("Balcony","#FFF200")
PaintToolObj.AddButton("Garage","#FFAEC9")

PaintToolObj.Run()
sys.exit("exit")"""

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image
  
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image
  
  
# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]
 
@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 532, 532)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image
  
  
  
  
"""plt.figure(figsize=(6, 6))
for i in range(4):
  rj_inp, rj_re = random_jitter(inp, re)
  plt.subplot(2, 2, i+1)
  plt.imshow(rj_inp/255.0)
  plt.axis('off')
plt.show()"""
  
def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image
  
def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image




train_dataset = tf.data.Dataset.list_files(DATBASE_PATH+'/train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
  
  
test_dataset = tf.data.Dataset.list_files(DATBASE_PATH+'/test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


testimage=Image.new("RGB",(512,512))
testimage.save('./Test/test0.png')
testimage.save('./Test/Result.png')
custom_test_dataset = tf.data.Dataset.list_files('./Test/test*')
custom_test_dataset = custom_test_dataset.map(load_image_test)
custom_test_dataset = custom_test_dataset.batch(BATCH_SIZE)



def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


#down_model = downsample(3, 4)
#down_result = down_model(tf.expand_dims(inp, 0))
#print (down_result.shape)


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
  
  
def Generator():
  inputs = tf.keras.layers.Input(shape=[512,512,3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
  
  
  
  
generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
  
  
#gen_output = generator(inp[tf.newaxis,...], training=False)
#plt.imshow(gen_output[0,...])


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss
  
  
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[512, 512, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[512, 512, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
  down3 = downsample(512, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)
  
  
  
discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
  
  
#disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
#plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
#plt.colorbar()
  
  
  
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss
  
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  
  
  
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


if RESTORE_CKP:  
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)) 

def generate_result_image(model, test_input, tar,Num):
  prediction = model(test_input, training=False)
  fig=plt.figure(figsize=(5.12,5.12),frameon=False)

 
  plt.imshow(prediction[0] * 0.5 + 0.5)
  plt.axis('off')
  plt.savefig('./Test/Result.png',bbox_inches='tight', pad_inches=0)
  plt.savefig('./Test/Drw_'+str(Num)+'.png',bbox_inches='tight', pad_inches=0)
  fig.clear()
  plt.close(fig)
  




if Open_Interactive_Tool:
	PaintToolObj=PaintTool()
	PaintToolObj.AddButton("Walls","#FFFFFF")
	#PaintToolObj.AddButton("Brick Walls","#FF7F27")
	PaintToolObj.AddButton("Dark Walls","#7F7F7F")
	PaintToolObj.AddButton("Woods Walls","#880015")
	PaintToolObj.AddButton("Window","#FF1C24")
	PaintToolObj.AddButton("Glass","#A349A4")
	PaintToolObj.AddButton("Balcony","#C8BFE7")
	PaintToolObj.AddButton("Tree","#B5E61D")
	PaintToolObj.AddButton("Grass","#22B14C")
	PaintToolObj.AddButton("Pool","#3F48CC")
	PaintToolObj.AddButton("Sky","#00A2E8")
	
	PaintToolObj.SetGenerator(generator)
	print("Painter Engine Running")
	PaintToolObj.Run()
	print("Stops Running!!!")
	sys.exit("exit")



def generate_images(model, test_input, tar,epoch):
  prediction = model(test_input, training=True)
  fig=plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig(OUTPUT_DIR+'/frame/image_at_epoch_{:04d}.png'.format(epoch),bbox_inches='tight', pad_inches=0)
  fig.clear()
  plt.close(fig)
  
  
for example_input, example_target in test_dataset.take(1):
  generate_images(generator, example_input, example_target,0)
  
  

  
  
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  
@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    #display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target,epoch+1+CURRENT_EPOCH)
    print("Epoch: ", epoch+1+CURRENT_EPOCH)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % CKP_SAVE_INT == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1+CURRENT_EPOCH,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)
  



fit(train_dataset, EPOCHS, test_dataset)

