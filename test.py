import numpy as numpy
import pandas as pd
import os
import pickle

def getFeatures(data):
	feats = []
	for i in range(0,len(data)):
		instance = data['feature'][i].tolist()
		feats.append(instance)
	feature  =  pd.DataFrame([subject for subject in feats])
	return feature

df = pd.read_hdf("./features/mfcc/timit_test.hdf")
df_d = pd.read_hdf("./features/mfcc_delta/timit_test.hdf")
df_dd = pd.read_hdf("./features/mfcc_delta_delta/timit_test.hdf")
t_mfcc = getFeatures(df)
t_mfcc_labels = df['labels']
t_mfcc_region = df['region']
t_mfcc_speaker = df['speaker']
t_mfcc_sentence = df['sentence']
t_mfcc_delta = getFeatures(df_d)
t_mfcc_delta_labels = df_d['labels']
t_mfcc_delta_delta = getFeatures(df_dd)
t_mfcc_delta_delta_labels = df_dd['labels']

def classify(models, models_w, test_data, column):
	person = {}
	result_mixture = {}
	print("Using Energy Coefficients\n")
	for phoneme in unique_labels:
		scores = (models[phoneme]).score_samples(test_data)
		result_mixture[phoneme] = scores

	for i in range(0,len(result_mixture['uh'])):
		max_label = 'sil'
		max_prob = -float('inf')
		for k in unique_labels:
			if(max_prob < (result_mixture[k])[i]):
				max_prob = (result_mixture[k])[i]
				max_label = k
		file = t_mfcc_region[i]+"_"+t_mfcc_speaker[i]+"_"+t_mfcc_sentence[i]
		if(max_label == t_mfcc_labels[i]):
			
			if(file in person):
				(person[file])['count'] = (person[file])['count']+1
			else:
				person[file] = {}
				(person[file])['count'] = 1
				(person[file])['total'] = 0

		elif(max_label == 'space' and t_mfcc_labels[i] == ''):

			if(file in person):
				(person[file])['count'] = (person[file])['count']+1

			else:
				person[file] = {}
				(person[file])['count'] = 1
				(person[file])['total'] = 0

		if(file in person):
			(person[file])['total'] = (person[file])['total']+1
		else:
			person[file] = {}
			(person[file])['total'] = 1
			(person[file])['count'] = 0
	count = 0
	total = 0
	for sen in person:
		accuracy = (person[sen])['count']/(person[sen])['total']
		accuracy = 1-accuracy
		#print(str(sen) + "---->"+(str(accuracy)))
		count = count+(person[sen])['count']
		total = total+(person[sen])['total']

	print("Accuracy = ", str(count/total))
	print("Without using Energy Coefficients\n")
	result_mixture = {}
	person = {}
	for phoneme in unique_labels:
		scores = (models_w[phoneme]).score_samples(test_data.drop(columns = column,axis = 1))
		result_mixture[phoneme] = scores
	for i in range(0,len(result_mixture['uh'])):
		max_label = 'sil'
		max_prob = -float('inf')
		for k in unique_labels:
			if(max_prob<(result_mixture[k])[i]):
				max_prob = (result_mixture[k])[i]
				max_label = k
		file = t_mfcc_region[i]+"_"+t_mfcc_speaker[i]+"_"+t_mfcc_sentence[i]
		if(max_label == t_mfcc_labels[i]):
			
			if(file in person):
				(person[file])['count'] = (person[file])['count']+1
			else:
				person[file] = {}
				(person[file])['count'] = 1
				(person[file])['total'] = 0

		elif(max_label == 'space' and t_mfcc_labels[i] == ''):

			if(file in person):
				(person[file])['count'] = (person[file])['count']+1

			else:
				person[file] = {}
				(person[file])['count'] = 1
				(person[file])['total'] = 0

		if(file in person):
			(person[file])['total'] = (person[file])['total']+1
		else:
			person[file] = {}
			(person[file])['total'] = 1
			(person[file])['count'] = 0
	count = 0
	total = 0
	for sen in person:
		accuracy = (person[sen])['count']/(person[sen])['total']
		accuracy = 1-accuracy
		#print(str(sen) + "---->"+(str(accuracy)))
		count = count+(person[sen])['count']
		total = total+(person[sen])['total']

	print("Frame Accuracy------->", str(count/total))


mfcc_models = {}
mfcc_models_w = {}
mfcc_delta_models = {}
mfcc_delta_models_w = {}
mfcc_delta_delta_models = {}
mfcc_delta_delta_models_w = {}

lns  =  open("unique_labels.txt",'r')
unique_labels = []

for line in lns:
	mfcc_models[(line.split('\n'))[0]] = {}
	mfcc_models_w[(line.split('\n'))[0]] = {}
	unique_labels.append((line.split('\n'))[0])


for phoneme in unique_labels:
	(mfcc_models[phoneme])['2'] = pickle.load(open("./models/"+phoneme+"/ec_2.pkl", 'rb'))
	(mfcc_models[phoneme])['4'] = pickle.load(open("./models/"+phoneme+"/ec_4.pkl", 'rb'))
	(mfcc_models[phoneme])['8'] = pickle.load(open("./models/"+phoneme+"/ec_8.pkl", 'rb'))
	(mfcc_models[phoneme])['16'] = pickle.load(open("./models/"+phoneme+"/ec_16.pkl", 'rb'))
	(mfcc_models[phoneme])['32'] = pickle.load(open("./models/"+phoneme+"/ec_32.pkl", 'rb'))
	(mfcc_models[phoneme])['64'] = pickle.load(open("./models/"+phoneme+"/ec_64.pkl", 'rb'))
	(mfcc_models[phoneme])['128'] = pickle.load(open("./models/"+phoneme+"/ec_128.pkl", 'rb'))
	(mfcc_models[phoneme])['256'] = pickle.load(open("./models/"+phoneme+"/ec_256.pkl", 'rb'))

	(mfcc_models_w[phoneme])['2'] = pickle.load(open("./models/"+phoneme+"/wec_2.pkl", 'rb'))
	(mfcc_models_w[phoneme])['4'] = pickle.load(open("./models/"+phoneme+"/wec_4.pkl", 'rb'))
	(mfcc_models_w[phoneme])['8'] = pickle.load(open("./models/"+phoneme+"/wec_8.pkl", 'rb'))
	(mfcc_models_w[phoneme])['16'] = pickle.load(open("./models/"+phoneme+"/wec_16.pkl", 'rb'))
	(mfcc_models_w[phoneme])['32'] = pickle.load(open("./models/"+phoneme+"/wec_32.pkl", 'rb'))
	(mfcc_models_w[phoneme])['64'] = pickle.load(open("./models/"+phoneme+"/wec_64.pkl", 'rb'))
	(mfcc_models_w[phoneme])['128'] = pickle.load(open("./models/"+phoneme+"/wec_128.pkl", 'rb'))
	(mfcc_models_w[phoneme])['256'] = pickle.load(open("./models/"+phoneme+"/wec_256.pkl", 'rb'))
	
	mfcc_delta_models[phoneme] = pickle.load(open("./models/"+phoneme+"/ec_delta.pkl", 'rb'))
	mfcc_delta_models_w[phoneme] = pickle.load(open("./models/"+phoneme+"/wec_delta.pkl", 'rb'))

	mfcc_delta_delta_models[phoneme] = pickle.load(open("./models/"+phoneme+"/ec_delta_delta.pkl", 'rb'))
	mfcc_delta_delta_models_w[phoneme] = pickle.load(open("./models/"+phoneme+"/wec_delta_delta.pkl", 'rb'))

mixture_list = ['2','4','8','16','32','64','128','256']
for mixture in mixture_list:
	print(str(mixture)+"\n")
	print("With Energy Coefficients\n")
	result_mixture = {}
	person = {}
	for phoneme in unique_labels:
		scores = (mfcc_models[phoneme])[mixture].score_samples(t_mfcc)
		result_mixture[phoneme] = scores
	for i in range(0,len(result_mixture['uh'])):
		max_label = 'sil'
		max_prob = -float('inf')
		for k in unique_labels:
			if(max_prob<(result_mixture[k])[i]):
				max_prob = (result_mixture[k])[i]
				max_label = k
		file = t_mfcc_region[i]+"_"+t_mfcc_speaker[i]+"_"+t_mfcc_sentence[i]
		if(max_label == t_mfcc_labels[i]):
			
			if(file in person):
				(person[file])['count'] = (person[file])['count']+1
			else:
				person[file] = {}
				(person[file])['count'] = 1
				(person[file])['total'] = 0

		elif(max_label == 'space' and t_mfcc_labels[i] == ''):

			if(file in person):
				(person[file])['count'] = (person[file])['count']+1

			else:
				person[file] = {}
				(person[file])['count'] = 1
				(person[file])['total'] = 0

		if(file in person):
			(person[file])['total'] = (person[file])['total']+1
		else:
			person[file] = {}
			(person[file])['total'] = 1
			(person[file])['count'] = 0
	count = 0
	total = 0
	for sen in person:
		accuracy = (person[sen])['count']/(person[sen])['total']
		accuracy = 1-accuracy
		#print(str(sen) + "---->"+(str(accuracy)))
		count = count+(person[sen])['count']
		total = total+(person[sen])['total']

	print("Frame Accuracy------->", str(count/total))
	print("Without Energy Coefficients\n")
	result_mixture = {}
	person = {}
	for phoneme in unique_labels:
		scores = (mfcc_models_w[phoneme])[mixture].score_samples(t_mfcc.drop(columns = [0],axis = 1))
		result_mixture[phoneme] = scores
	for i in range(0,len(result_mixture['uh'])):
		max_label = 'sil'
		max_prob = -float('inf')
		for k in unique_labels:
			if(max_prob<(result_mixture[k])[i]):
				max_prob = (result_mixture[k])[i]
				max_label = k
		file = t_mfcc_region[i]+"_"+t_mfcc_speaker[i]+"_"+t_mfcc_sentence[i]
		if(max_label == t_mfcc_labels[i]):
			
			if(file in person):
				(person[file])['count'] = (person[file])['count']+1
			else:
				person[file] = {}
				(person[file])['count'] = 1
				(person[file])['total'] = 0

		elif(max_label == 'space' and t_mfcc_labels[i] == ''):

			if(file in person):
				(person[file])['count'] = (person[file])['count']+1

			else:
				person[file] = {}
				(person[file])['count'] = 1
				(person[file])['total'] = 0

		if(file in person):
			(person[file])['total'] = (person[file])['total']+1
		else:
			person[file] = {}
			(person[file])['total'] = 1
			(person[file])['count'] = 0
	count = 0
	total = 0
	for sen in person:
		accuracy = (person[sen])['count']/(person[sen])['total']
		accuracy = 1-accuracy
		#print(str(sen) + "---->"+(str(accuracy)))
		count = count+(person[sen])['count']
		total = total+(person[sen])['total']

	print("Frame Accuracy------->", str(count/total))

classify(mfcc_delta_models, mfcc_delta_models_w,t_mfcc_delta,[0,13])
classify(mfcc_delta_delta_models, mfcc_delta_delta_models_w,[0,13,26])