import tensorflow as tf
import numpy as np
import keras.backend as K
from convert_result import convert_result

anchors_value = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]],dtype='float32')
nb_classes = 80
anchors_length = len(anchors_value)

#used for final result

def loss_function(args):
	#should work fun
	"""
	y_true (batch, 13, 13, 425) tensor
	y1 (batch, 5) tensor
	y2 (batch, 13,13,5,1)
	y3 (batch, 13,13,5,5)
	"""
	y_pred, y1, y2, y3 = args
	#converted_result = convert_result(y_pred, anchors, nb_classes)
	#mask = tf.greater_equal(y1, 0)
	#true_boxes = tf.boolean_mask(y1, mask)
	return loss_calculator(y_pred, y1, y2, y3)


def loss_calculator(output, true_boxes, object_mask, object_value):
	"""
	calculate loss on the basis of a batch
	para:
		output: output by the net. (13, 13, 485)
		true_boxes: shape (5*k,). x,y is the top left corner. may need to change
		object_mask: shape(batch_size, 13, 13, 5, 1), with entry equals 1 means this anchor is the 
			right one. 1obj in loss equation
		object_value: shape(batch_size, 13, 13, 5, 5), indicates the x, y, w, h and class for the 
			right box
	"""

	#use convert_result to convert output. bxy is bx, by. 
	bxy, bwh, to, classes = convert_result(output, anchors_value, nb_classes)

	#leave the ratio unassigned right now
	alpha1 = 5.0
	alpha2 = 5.0
	alpha3 = 1.0
	alpha4 = 0.1
	alpha5 = 1

	#first term coordinate_loss
	#bxy_sigmoid = bxy - tf.floor(bxy)
	bxy_loss = K.sum(K.square(bxy - object_value[...,0:2])*object_mask)

	#second term
	bwh_loss = K.sum(K.square(K.sqrt(bwh)-K.sqrt(object_value[...,2:4]))*object_mask)

	#third term
	to_obj_loss = K.sum(K.square(1-to)*object_mask)

	#forth term. TODO, need to multiply another factor.  (1 - object_detection)
	#calculate object_detection
	output_shape = tf.shape(output)
	bxy = tf.expand_dims(bxy, 4)#(batch,13, 13, 5, 1, 2)
	bwh = tf.expand_dims(bwh, 4)#(batch, 13, 13, 5, 1, 2)
	wh_half = bwh / 2.
	pred_mins = bxy - wh_half#(batch,13, 13, 5, 1, 2)
	pred_maxes = bxy + wh_half#(batch,13, 13, 5, 1, 2)

	true_boxes_shape = tf.shape(true_boxes)
	true_boxes = tf.reshape(true_boxes, (true_boxes_shape[0], 1, 1, 1, -1, 5))
	#true_boxes(batch, 1, 1, 1, 10, 5)
	true_mins = true_boxes[...,:2] - true_boxes[...,2:4]/2.#(batch,1, 1, 1, 10, 2)
	true_maxes = true_boxes[...,:2] + true_boxes[...,2:4]/2.#(batch,1, 1, 1, 10, 2)

	intersect_mins = K.maximum(true_mins, pred_mins)#(batch, 13, 13, 5, 10, 2)
	intersect_maxes = K.minimum(true_maxes, pred_maxes)#(batch,13, 13, 5, 10, 2)
	intersect_wh = K.maximum(intersect_maxes-intersect_mins, 0)#(batch,13, 13, 5, 10, 2)
	intersect_areas = intersect_wh[...,0]*intersect_wh[...,1]#(batch,13, 13, 5, 10)

	pred_areas = bwh[...,0]*bwh[...,1]
	true_areas = true_boxes[...,2]*true_boxes[...,3]#(batch,13, 13, 5, 10)
	#iou_score:(batch, 13,13,5,10)
	iou_score = intersect_areas/(pred_areas + true_areas - intersect_areas)#(batch,13, 13, 5, 10)
	mask = K.greater(K.sum(true_boxes,axis=5),0)#(batch, 1,1,1,10)
	iou_score = iou_score*(K.cast(mask, dtype=K.dtype(true_boxes)))
	best_iou = K.max(iou_score, axis=4)#(batch, 13, 13, 5)

	best_iou = K.expand_dims(best_iou)#(batch,13,13,5,1)

	object_detection = K.cast(best_iou>0.6,K.dtype(best_iou))#(batch,13,13,5,1)

	#object_mask:(batch_size, 13, 13, 5, 1)
	#this loss happens only when no true object but detect object. I guess object_detection is
	#	not very important as long as you lower alpha4. This term will push to to 0 for every 
	#	position. I guess just use random mask may also works
	to_noobj_loss = K.sum(K.square(0-to)*(1-object_mask)*object_detection)

	#fifth term
	onehot_class = K.one_hot(tf.to_int32(object_value[...,4]), nb_classes)
	class_loss = K.sum(K.square(onehot_class-classes)*object_mask)

	#total loss
	result = alpha1*bxy_loss + alpha2*bwh_loss + alpha3*to_obj_loss + \
			alpha4*to_noobj_loss + alpha5*class_loss

	return result

