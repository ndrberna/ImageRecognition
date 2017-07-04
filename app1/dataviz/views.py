from django.http import HttpResponse
import sys
sys.path.append("..")
import graphlab
from django.shortcuts import render_to_response, render, redirect
from django.template.context import RequestContext
from django.http import HttpResponse, HttpResponseRedirect
from time import sleep
import requests
import os

'''
Stage 1: Train a DNN classifier on a large, general dataset. A good example is ImageNet ,with 1000 categories and 1.2 million images. GraphLab hosts a model trained on ImageNet to allow you to skip this step in your own implementation. Simply load the model with
gl.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')

Stage 2: The outputs of each layer in the DNN can be viewed as a meaningful vector representaion of each image. Extract these feature vectors from the layer prior to the output layer on each image of your task. 
Stage 3: Train a new classifier with those features as input for your own task.  
'''



cifarLabels=["airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"]


# versione piccola del cifar
'''
image_train = graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/coursera/deep_learning/image_train_data')
image_test = graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/coursera/deep_learning/image_test_data')
deep_features_model = graphlab.neuralnet_classifier.create(image_train, features=['deep_features'], target='label',verbose=False)
deep_features_model_logistic = graphlab.logistic_classifier.create(image_train, features=['deep_features'], target='label',verbose=False)
deep_learning_model_extractor = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
'''
root='/Users/thevault/.virtualenvs/demonstrator2018/app1/data'
image_train=graphlab.load_sframe(root+'/image_train_data')
image_test=graphlab.load_sframe(root+'/image_test_data')
deep_model=graphlab.load_model(root+"/deep_model")
raw_pixel_model=graphlab.load_model(root+"/raw_pixel_model")
deep_features_model_logistic=graphlab.load_model(root+"/deep_features_model_logistic")
deep_learning_model_extractor=graphlab.load_model(root+"/deep_learning_model_extractor")



cats_dogs_deep_features_model=graphlab.load_model(root+"/cats_dogs_deep_features_model")
brand_deep_features_model=graphlab.load_model(root+"/brand_deep_features_model")


food_boosted_tree_model=graphlab.load_model(root+"/food_boosted_tree_model")

fashion_classifier=graphlab.load_model(root+"/fashion_classifier")

'''
deep_learning_model_extractor = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')

# versione grande del cifar
image_train = graphlab.SFrame('http://s3.amazonaws.com/GraphLab-Datasets/cifar_10/cifar_10_train_sframe')
image_test = graphlab.SFrame('http://s3.amazonaws.com/GraphLab-Datasets/cifar_10/cifar_10_test_sframe')
#image_train['image'] = graphlab.image_analysis.resize(image_train['image'], 256, 256, 3)
#image_test['image'] = graphlab.image_analysis.resize(image_test['image'], 256, 256, 3)

image_train['deep_features'] = deep_learning_model_extractor.extract_features(image_train)
image_test['deep_features'] = deep_learning_model_extractor.extract_features(image_test)

deep_features_model = graphlab.neuralnet_classifier.create(image_train, features=['deep_features'], target='label',verbose=False)

'''



def viz(request):

	return render_to_response('search.html')

def cat_dog(request):

	#file access
	url = request.GET.get('url')
	
	print os.getcwd()
	path='image/image_name.jpg'

	try:
		img_data = requests.get(url).content

		with open(path, 'wb') as handler:
		    handler.write(img_data)

		img_sframe=graphlab.image_analysis.load_images(os.getcwd()+"/image/", format='auto', with_path=True, recursive=True, ignore_failure=True, random_order=True)
		print img_sframe

		img_sframe['features'] = deep_learning_model_extractor.extract_features(img_sframe) 
		result=cats_dogs_deep_features_model.classify(img_sframe)
		result_tag=["cat","dog"]
		probabilita=result['probability']
		result=result_tag[result['class'][0]]
		success=True
		
		context = {
		'result':result,
		'probabilita':probabilita,
		'image':url,
		'success':success
		}

	except:
		success=False
		context = {
		'url':url,
		'success':success

		}
	return render_to_response('cat&dog.html',context)
	

def brand(request):
	url = request.GET.get('url')
		
	print os.getcwd()
	path='image/image_name.jpg'
	
	try:
		
		img_data = requests.get(url).content

		with open(path, 'wb') as handler:
		    handler.write(img_data)

		img_sframe=graphlab.image_analysis.load_images(os.getcwd()+"/image/", format='auto', with_path=True, recursive=True, ignore_failure=True, random_order=True)
		print img_sframe

		#img_sframe['features'] = deep_learning_model_extractor.extract_features(img_sframe) 
		img_sframe['deep_features'] = deep_learning_model_extractor.extract_features(img_sframe)


		result=brand_deep_features_model.classify(img_sframe)
		print "RISULTATO",result
		
		result_tag=["rbb","LA7","", "RTSun","ARD1","FR3","ZDF","TVE2","YLE","la7do","BBCTHREE","RAINEW","rbb","","TV2000","WDR","C5","NFX","BR"]
		probabilita=result['probability']
		result=result_tag[result['class'][0]]
		success=True
		context = {
		'result':result,
		'probabilita':probabilita,
		'image':url,
		'success':success
		}


	except:
		success=False
		context = {
		'url':url,
		'success':success

		}


	return render_to_response('brand.html',context)
	


def scrape(request):
	
	url = request.GET.get('url')
	
	print os.getcwd()
	path='image/image_name.jpg'
	
	try:
		img_data = requests.get(url).content

		with open(path, 'wb') as handler:
		    handler.write(img_data)

		img_sframe=graphlab.image_analysis.load_images(os.getcwd()+"/image/", format='auto', with_path=True, recursive=True, ignore_failure=True, random_order=True)
		print img_sframe

		img_sframe['deep_features'] = deep_learning_model_extractor.extract_features(img_sframe) 

		probabilita=deep_features_model_logistic.classify(img_sframe)['probability']
		result=deep_features_model_logistic.classify(img_sframe)['class'][0]
		success=True
		context = {
		'result':result,
		'probabilita':probabilita,
		'image':url,
		'success':success
		}


	except:
		success=False
		context = {
		'url':url,
		'success':success

		}
	return render_to_response('demo1_result.html',context)
	


def food(request):

	url = request.GET.get('url')
		
	print os.getcwd()
	path='image/image_name.jpg'
	
	try:
		
		img_data = requests.get(url).content

		with open(path, 'wb') as handler:
		    handler.write(img_data)

		img_sframe=graphlab.image_analysis.load_images(os.getcwd()+"/image/", format='auto', with_path=True, recursive=True, ignore_failure=True, random_order=True)
		print img_sframe

		
		img_sframe['deep_features'] = deep_learning_model_extractor.extract_features(img_sframe)


		result=food_boosted_tree_model.classify(img_sframe)
		print "RISULTATO",result
		
		probabilita=result['probability'][0]
		result=result['class'][0]
		success=True
		context = {
		'result':result,
		'probabilita':probabilita,
		'image':url,
		'success':success
		}


	except:
		success=False
		context = {
		'url':url,
		'success':success

		}


	return render_to_response('food.html',context)
	


def fashion(request):

	url = request.GET.get('url')
		
	print os.getcwd()
	path='image/image_name.jpg'
	
	try:
		
		img_data = requests.get(url).content

		with open(path, 'wb') as handler:
		    handler.write(img_data)

		img_sframe=graphlab.image_analysis.load_images(os.getcwd()+"/image/", format='auto', with_path=True, recursive=True, ignore_failure=True, random_order=True)
		print img_sframe

		
		img_sframe['deep_features'] = deep_learning_model_extractor.extract_features(img_sframe)


		result=fashion_classifier.classify(img_sframe)
		print "RISULTATO",result
		
		probabilita=result['probability'][0]
		result=result['class'][0]
		success=True
		context = {
		'result':result,
		'probabilita':probabilita,
		'image':url,
		'success':success
		}


	except:
		success=False
		context = {
		'url':url,
		'success':success

		}


	return render_to_response('fashion.html',context)
	

