############################################################################
#                                                                          #
#Sample Detector v0.1 By Guo Ruocheng: mcspinemo@gmail.com                 #
#3 basic features and 2 optional features included                         #
#                                                                          #
#Please run it with python 2.7                                             #
#Please modify the path before execute                                     #
#Manual decision boundary included for generate target value of train data #
#Result would be recorded pkl file                                         #
#                                                                          #
############################################################################





from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from sklearn import svm
from sklearn.externals import joblib


def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('/home/mcspinemo_guo/Downloads/Albert Wong/train.csv', 'r'),skip_header = True, delimiter = ',')
    testdataset = genfromtxt(open('/home/mcspinemo_guo/Downloads/Albert Wong/test.csv', 'r'),skip_header = True, delimiter=',')
    target = [x[6] for x in dataset]
    train = [x[1:6] for x in dataset]
    
    #create and train the random forest
    #multi-core CPU version
    

    
    test = [x[1:6] for x in testdataset]
    
    

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    result = rf.predict(test)
    
    #result = clf.predict(test)
    for i in result:
        print i
    
    
    import cPickle
    # save the classifier
    with open('/home/mcspinemo_guo/Downloads/Albert Wong/my_dumped_classifier.pkl', 'wb') as fid:
        cPickle.dump(rf, fid)    
    
    # load it again
    with open('/home/mcspinemo_guo/Downloads/Albert Wong/my_dumped_classifier.pkl', 'rb') as fid:
        myrfloaded = cPickle.load(fid)
    
    print myrfloaded.predict(test)
    

if __name__=="__main__":
    main()

