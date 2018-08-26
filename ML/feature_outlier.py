import pandas as pd  
import numpy as np

#设置随机数种子
np.random.seed(0)

def isolation_del_outlier(X):

	from sklearn.ensemble import IsolationForest
	#孤立森联删除离群值，假设离群比列0.05,训练样本数比列为0.1
	isolationForest=IsolationForest(n_estimators =100,contamination=0.1,max_samples=0.1)
	isolationForest.fit(X)
	isolationLabel=isolationForest.predict(X) 
	return X[isolationLabel==1]

def isolationLabel_del_outlier(X):

	from sklearn.ensemble import IsolationForest
	#孤立森联删除离群值，假设离群比列0.05,训练样本数比列为0.1
	isolationForest=IsolationForest(n_estimators =100,contamination=0.1,max_samples=0.1)
	isolationForest.fit(X)
	isolationLabel=pd.Series(isolationForest.predict(X) )
	return isolationLabel

def load_data(feature_list):
	print('loading data...')
	train=pd.concat([pd.read_csv('file//train_'+feature+'.csv',index_col =0) for feature in feature_list ],ignore_index=True,axis=1)
	test=pd.concat([pd.read_csv('file//test_'+feature+'.csv',index_col =0) for feature in feature_list ],ignore_index=True,axis=1)
	train_y=pd.read_csv('file//train_y.csv',index_col=0)
	
	return train,train_y,test

def load_data_separation(feature):
	print(f'loading {feature} data...')
	train=pd.read_csv('file//train_'+feature+'.csv',index_col =0)
	test=pd.read_csv('file//test_'+feature+'.csv',index_col =0) 
	train_y=pd.read_csv('file//train_y.csv',index_col=0)
	
	return train,train_y,test


def del_outlier(train,train_y,del_outlier_mode):
	print('delete outlier...')
	print('del_outlier_mode:  ',del_outlier_mode)
	print('train samples',train.shape[0])
	if del_outlier_mode=='each_class':
		train=pd.concat([train,train_y],axis=1).groupby('class',as_index=False).apply(isolation_del_outlier).reset_index(drop=True)
		train_y=train['class']
		train=train.drop(['class'],axis=1)
	elif del_outlier_mode=='all':
		train,train_y=isolation_del_outlier(pd.concat([train,train_y],axis=1))
	else:
		pass
	print('new_train samples',train.shape[0])
	return train,train_y

def del_outlier_label(train,train_y,del_outlier_mode):

	if del_outlier_mode=='each_class':
		isolationLabel=pd.concat([train,train_y],axis=1).groupby('class',as_index=False).apply(isolationLabel_del_outlier).reset_index(drop=True)

	elif del_outlier_mode=='all':
		isolationLabel=isolationLabel_del_outlier(pd.concat([train,train_y],axis=1))
	else:
		isolationLabel=[1]*train.shape[0]
	return isolationLabel


def selecter_features(train,train_y,test,feature_selector,normalize=True,threshold=1e-5,n_features_to_select=0.8):
	print('lsvc prefit...')
	print('feature_selector:  ',feature_selector)
	print('train feature dimension',train.shape[1]-1)
	from sklearn.svm import LinearSVC
	lsvc = LinearSVC(class_weight='balanced',penalty="l1", dual=False)

	print('selecting features...')

	if feature_selector=='SelectFromModel':
		from  sklearn.feature_selection import SelectFromModel
		lsvc.fit(train, train_y)
		train= pd.DataFrame(SelectFromModel(lsvc,threshold=threshold, prefit=True).transform(train))
		test= pd.DataFrame(SelectFromModel(lsvc,threshold=threshold, prefit=True).transform(test))
	elif feature_selector=='RFE' :
		from  sklearn.feature_selection import RFE
		ref=RFE(lsvc,n_features_to_select=int(n_features_to_select*train.shape[1]))
		print(int(n_features_to_select*train.shape[1]))
		train=pd.DataFrame(ref.fit_transform(train, train_y))
		test=pd.DataFrame(ref.transform(test))
	else:
		pass

	if  normalize ==True:
		from  sklearn.preprocessing import MinMaxScaler
		scaler = MinMaxScaler()
		train=pd.DataFrame(scaler.fit_transform(train))
		test=pd.DataFrame(scaler.fit_transform(test))

	print('new_train feature dimension',train.shape[1]-1)
	return train,train_y,test

def featuere_stacking(feature_list,del_outlier_mode):

	train,train_y,test=load_data(feature_list)
	isolationLabel=del_outlier_label(train,train_y,del_outlier_mode)

	from sklearn.model_selection import StratifiedKFold
	dataset_blend_train = np.zeros((sum(isolationLabel==1), len(feature_list)))
	dataset_blend_test = np.zeros((np.array(test).shape[0], len(feature_list)))
	ensemble_clf=ensemble_model(clf_list,meta_classifier,mode='stacking')
	for j,feature in enumerate(feature_list):

		train,train_y,test=load_data_separation(feature)
		train,train_y=train[isolationLabel==1],train_y[isolationLabel==1]
		train,train_y,test=selecter_features(train,train_y,test,feature_selector,threshold,n_features_to_select)
		train,train_y,test=np.array(train),np.array(train_y),np.array(test)
		skf = list(StratifiedKFold(n_splits=5).split(train, train_y))
		dataset_blend_test_j = np.zeros((test.shape[0], len(skf)))

		for i, (train_index, test_index) in enumerate(skf):
			'''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
			X_train, y_train, X_test, y_test = train[train_index], train_y[train_index], train[test_index], train_y[test_index]
			ensemble_clf.fit(X_train, y_train)
			y_submission = ensemble_clf.predict(X_test)
			dataset_blend_train[test_index, j] = y_submission
			dataset_blend_test_j[:, i] = ensemble_clf.predict(test)
			'''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
		dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

	return dataset_blend_train,train_y,dataset_blend_test


if __name__ =='__main__':
	feature_list=['hdp','rp']         #
	del_outlier_mode='each_class'
	feature_selector='RFE'
	threshold=1e-5
	n_features_to_select=0.8

	train,train_y,test=load_data(feature_list)
	train,train_y=del_outlier(train,train_y,del_outlier_mode)
	train,train_y,test=selecter_features(train,train_y,feature_selector,threshold,n_features_to_select)


	print('saving new_train,new_test...')
	pd.concat([train,train_y],axis=1).to_csv('file\\new_train.csv')
	test.to_csv('file\\new_test.csv')
