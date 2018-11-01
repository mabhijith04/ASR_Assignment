import pandas as pd
import numpy as np

df = pd.read_hdf("./features/mfcc/timit_train.hdf")
df_delta=pd.read_hdf("./features/mfcc_delta/timit_train.hdf")
df_dd=pd.read_hdf("./features/mfcc_delta_delta/timit_train.hdf")
print(df.head())
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
features_delta=np.array(df_delta["features"].tolist())
features_delta_labels=np.array(df_delta["labels"].tolist())
features_dd=np.array(df_dd["features"].tolist())
features_dd_labels=np.array(df_dd["labels"].tolist())
unique_labels=[]
for ite in labels:
	if(ite not in unique_labels):
		unique_labels.append(ite)

number_of_examples=len(labels)
separated={}
separated_d={}
separated_dd={}
for l in unique_labels:
	if(l==""):
		l="space"
	separated[l]=pd.DataFrame()
	separated_d[l]=pd.DataFrame()
	separated_dd[l]=pd.DataFrame()
for i in range(0,number_of_examples):
	if(labels[i]==""):
		labels[i]="space"
	if(features_delta_labels[i]==""):
		features_delta_labels[i]="space"
	if(features_dd_labels[i]==""):
		features_dd_labels[i]="space"
	s1=pd.Series(features[i])
	separated[labels[i]]=separated[labels[i]].append(s1,ignore_index=True)
	s2=pd.Series(features_delta[i])
	separated_d[features_delta_labels[i]] = separated_d[features_delta_labels[i]].append(s2,ignore_index=True)
	s3=pd.Series(features_dd[i])
	separated_dd[features_dd_labels[i]]=separated_dd[features_dd_labels[i]].append(s3,ignore_index=True)

for k in separated:
	fname=str(k)+"_train.hdf"
	separated[k].to_hdf("./features2/mfcc/"+fname, k)
	fname=str(k)+"_train.hdf"
	separated_d[k].to_hdf("./features2/mfcc_delta/"+fname,k)
	fname=str(k)+"_train.hdf"
	separated_dd[k].to_hdf("./features2/mfcc_delta_delta/"+fname,k)






lns=open("unique_labels.txt",'w')
for l in unique_labels:
	if(l==""):
		lns.write("space\n")
	else:
		lns.write(str(l)+"\n")

#Training code
