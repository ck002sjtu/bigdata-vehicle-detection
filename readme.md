#Please see demo here
https://www.youtube.com/watch?v=fOqRSUPtKFs&t=6s

#download weight
wget http://pjreddie.com/media/files/yolo.weights

#create a darknet model, which has 80 classes by default. This will generate a model.h5 file.
python model.py darknet -w yolo.weights -s model_data/model.h5
#use this model to do testing. There are some other parameters, you can ignore it to make it 		using default value
python detect_image.py model_data/model.h5

#detect video
python detect_video.py model_data/model.h5 -t /home/yh/Downloads/test.mp4 -o demo.avi


#fine tune customized model.
a. #create customized model which you can define the number of classes
	python model.py customized -l False -s model_data/customized.h5 -f 65
b. First you need to change nb_classes in loss_util.py
c. then run script in train_model.ipyth. 
	(1)Change parameters in model_to_train to fit your need first para is where to load the model. sencond is what optimizer to use. More details can be found at train_model.py. 
	(2)Change parameters in train_model. The second one is where to read data(it should be a txt file, in which first column is where to read this image, the following values are repeated x,y,w,h,class. The x,y should be pixel from top left corner, and w,h also in pixel). Another parameter you may change is nb_classes. And if you change nb_classes, you also need to change nb_classes in utils/loss_util.py. Other parameters' meaning are suggested by their name.
	Then you can find you fine-tuned model in save_path
d. #use this fine-tuned model to predict
	python detect_image.py model_data/customized.h5 -w model_data/new_model.h5 -c model_data/\ customized_classes.txt
