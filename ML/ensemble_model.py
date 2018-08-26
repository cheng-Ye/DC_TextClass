from feature_outlier import *
from sklearn.model_selection import cross_val_score
import pandas as pd

def SelectModel(modelname):

	if modelname == 'SVC':
		from sklearn.svm import SVC
		model = SVC(class_weight='balanced',probability=True)

	elif modelname == "LGBMClassifier":
		from lightgbm.sklearn import LGBMClassifier
		model = LGBMClassifier(class_weight='balanced',n_estimators=500,reg_alpha=0.1,colsample_bytree=0.8,subsample=0.8,verbose=-1)

	elif modelname == "RandomForestClassifier":
		from sklearn.ensemble import RandomForestClassifier
		model = RandomForestClassifier(n_estimators=100,class_weight='balanced')

	elif modelname == "XGBClassifier":
		from xgboost.sklearn import XGBClassifier
		model =  XGBClassifier(n_estimators=100,reg_alpha=0.1,colsample_bytree=0.8,subsample=0.8,class_weight='balanced')

	elif modelname == "KNeighborsClassifier":
		from sklearn.neighbors import KNeighborsClassifier 
		model = KNeighborsClassifier (weights='distance')

	elif modelname== 'GaussianNB':
		from sklearn.naive_bayes import GaussianNB 
		model=GaussianNB ()

	elif modelname== 'Rocchio':
		from sklearn.neighbors import NearestCentroid
		model=NearestCentroid()

	elif modelname== 'LogisticRegression':
		from sklearn.linear_model import LogisticRegression
		model=LogisticRegression(class_weight='balanced',penalty="l1")
	else:
		pass

	return model


def featuere_stacking(clf_list,meta_classifier,feature_list,del_outlier_mode,feature_selector,threshold,n_features_to_select):

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

def ensemble_model(clf_list,meta_classifier,mode='stacking',verbose=0):

	classifiers=[ SelectModel(clf)   for clf in clf_list]
	if mode=='stacking':
		from mlxtend.classifier import StackingClassifier
		ensemble_clf = StackingClassifier(classifiers=classifiers,use_probas=True,
								average_probas=False,meta_classifier=SelectModel(meta_classifier),verbose=verbose)

	else:
		from mlxtend.classifier  import EnsembleVoteClassifier
		ensemble_clf = EnsembleVoteClassifier(clfs=classifiers, voting='soft',verbose=verbose)

	return ensemble_clf


if __name__ =='__main__':

	clf_list=['SVC','LogisticRegression','KNeighborsClassifier','GaussianNB','XGBClassifier','RandomForestClassifier','Rocchio']

	meta_classifier='LogisticRegression'    

	train=pd.read_csv('file\\new_train.csv',index_col =0)
	train_y=train['class']
	train=train.drop(['class'],axis=1)
	ensemble_clf=ensemble_model(clf_list,meta_classifier,mode='stacking',verbose=1)
	print('k-fold cross validation:\n')
	#比赛评分指标 f1-score
	ensemble_clf.fit(train,train_y)

	#print(ensemble_clf.predict(train))

	scores = cross_val_score(ensemble_clf, train, train_y-1, cv=5, scoring='f1_weighted')  

	print('clf_list:',clf_list)
	print('\nmeta_classifier',meta_classifier)
	print("f1-score: %0.2f " % (scores.mean()))


