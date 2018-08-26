from preprocess_data import *
from feature_outlier import *
from ensemble_model import *

nrows=2500       #nrows不是参数，指的是用多少样本来调参，（越大越好，最大值为102277）电脑不行可适当减少

train_mode='separation'  #(对提取的特征选择训练方式)       'concatenation' or  'separation'

#preprocess_data     hyper-parameters                    
no_below=0.1             #(过滤掉单词出现在所有句子中的次数低于n_below的)    整数  ，范围2~10000
no_above=0.5           #(过滤掉单词出现在所有句子中的频率高于n_above的)    浮点数  ，范围0~1
num_topics=300         #  整数  ，范围200~500

#feature_outlier     hyper-parameters
feature_list=['hdp','rp']      #（选择使用哪些特征）feature_list 为 [hdp','rp','tfidf','log','lsi'] 的列表子集 比如feature_list=['hdp','rp']   
del_outlier_mode='each_class'   #  （三选一）  'each_class'  or 'all'  or None
feature_selector='SelectFromModel'   # （三选一）  'SelectFromModel'  or 'RFE'  or None
threshold=1e-5                       # 浮点数    （当feature_selector='SelectFromModel' ）选择特征重要性大于threshold的   范围0~1
n_features_to_select=0.8             # 浮点数    （当feature_selector='RFE' ）选择特征比率   范围0~1

#ensemble_model     hyper-parameters
'''（选择使用哪些模型）clf_list 为 ['LinearSVC','LogisticRegression','KNeighborsClassifier','RandomForestClassifier','XGBClassifier','GaussianNB'] 的列表子集 
 clf_list=['LinearSVC','LogisticRegression','KNeighborsClassifier']  ，子集一定要有'LinearSVC'  '''
clf_list=['SVC','LogisticRegression','KNeighborsClassifier','GaussianNB']
meta_classifier='LogisticRegression'   #(二选一) 'LogisticRegression' or 'LinearSVC' 


def run():
	import warnings
	warnings.filterwarnings("ignore")

	#设置随机数种子
	np.random.seed(0)
	now = datetime.now()

	#preprocess_data 
	train=pd.read_csv('..\\new_data\\train_set.csv',nrows=nrows, chunksize=500,usecols=['word_seg','class'])
	test=pd.read_csv('..\\new_data\\test_set.csv',nrows=nrows, chunksize=500,usecols=['word_seg'])
	no_below=int(0.1*nrows)
	gen_data(train,test,no_below,no_above,num_topics)
		#feature_outlier

	if train_mode=='separation' :
		dataset_blend_train,train_y,dataset_blend_test=featuere_stacking(clf_list,meta_classifier,feature_list,del_outlier_mode,feature_selector,threshold,n_features_to_select)


		print('k-fold cross validation:\n')
		print('clf_list:',clf_list)
		print('train_mode:',train_mode)

		model=ensemble_model(clf_list,meta_classifier,mode='voting')
		scores = cross_val_score(model, dataset_blend_train, train_y-1, cv=5, scoring='f1_weighted')
		print("voting f1-score: %0.2f " % (scores.mean()))

	elif train_mode=='concatenation':

		train,train_y,test=load_data(feature_list)
		train,train_y=del_outlier(train,train_y,del_outlier_mode)

		train,train_y,test=selecter_features(train,train_y,test,feature_selector,threshold,n_features_to_select)
		
		#print('saving new_train,new_test...')
		#pd.concat([train,train_y],axis=1).to_csv('file\\new_train.csv')

		#ensemble_model
		ensemble_clf=ensemble_model(clf_list,meta_classifier,mode='stacking')
		
		print('k-fold cross validation:\n')
		print('clf_list:',clf_list)
		print('train_mode:',train_mode)
		#比赛评分指标 f1-score
		scores = cross_val_score(ensemble_clf, train, train_y-1, cv=5, scoring='f1_weighted')  

		print('\nmeta_classifier',meta_classifier)
		print("f1-score: %0.2f " % (scores.mean()))


	print((datetime.now()-now))

if __name__ =='__main__':
	run()