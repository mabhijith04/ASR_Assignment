import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle


def getFeatures(data):
	feats=[]
	for i in range(0,len(data)):
		instance=data['feature'][i].tolist()
		feats.append(instance)
	feature = pd.DataFrame([subject for subject in feats])
	return feature

label=[]
lines=open("unique_labels.txt",'r')
for line in lines:
	label.append((line.split('\n'))[0])
	os.makedirs('models/'+line, exist_ok=True)

for phoneme in label:
	fname=str(phoneme)+"_train.hdf"
	df=pd.read_hdf("./features2/mfcc/"+fname)
	features=getFeatures(df)
	
	for mix in range(1,9):
		mixture=pow(2,mix)
		model_name="ec_"+str(mixture)+".pkl"
		model_name_w="wec_"+str(mixture)+".pkl"
		model=GaussianMixture(n_components=mixture, covariance_type='diag')
		model.fit(features.as_matrix(columns=None))
		model_w=GaussianMixture(n_components=mixture, covariance_type='diag')
		model_w.fit((features.drop(columns=[0],axis=1)).as_matrix(columns=None))
		pickle.dump(model,open("./models/"+str(phoneme)+"/"+model_name, 'wb'))
		pickle.dump(model_w, open("./models/"+str(phoneme)+"/"+model_name_w, 'wb'))

	df_d=pd.read_hdf("./features2/mfcc_delta/"+fname)
	features_d=getFeatures(df_d)
	df_dd=pd.read_hdf("./features2/mfcc_delta_delta/"+fname)
	features_dd=getFeatures(df_dd)
		
	model_d_name="ec_delta.pkl"
	model_d_name_w="wec_delta.pkl"
	model_dd_name="ec_delta_delta.pkl"
	model_dd_name_w="wec_delta_delta.pkl"

	model1=GaussianMixture(n_components=256, covariance_type='diag')
	model1.fit(features_d.as_matrix(columns=None))
	pickle.dump(model1, open("./models/"+str(phoneme)+"/"+model_d_name, 'wb'))
	
	model1_w=GaussianMixture(n_components=256,covariance_type='diag')
	model1_w.fit((features_d.drop(columns=[0,13],axis=1)).as_matrix(columns=None))
	pickle.dump(model1_w, open("./models/"+str(phoneme)+"/"+model_d_name_w, 'wb'))
	
	model2=GaussianMixture(n_components=256, covariance_type='diag')
	model2.fit(features_dd.as_matrix(columns=None))
	pickle.dump(model2, open("./models/"+str(phoneme)+"/"+model_dd_name, 'wb'))
	
	model2_w=GaussianMixture(n_components=256, covariance_type='diag')
	model2_w.fit((feature_dd.drop(columns=[0,13,26],axis=1)).as_matrix(columns=None))
	pickle.dump(model2_w, open("./models/"+str(phoneme)+"/"+model_dd_name_w, 'wb'))