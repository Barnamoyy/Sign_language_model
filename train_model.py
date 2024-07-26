# sklearn imports for random forest classifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import numpy as np 
import pickle 

# get the data and labels from the pickle file 
f = open('data.pickle', 'rb')

# data_dict because the pickle file is a key value pair dictionary 
data_dict = pickle.load(f)

# get the data 
data = np.asarray(data_dict['data'])

# get the labels 
labels = np.asarray(data_dict['labels'])
f.close()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# print("X_train shape: {}".format(X_train.shape)) >> X_train shape: (79, 42)
# print("y_train shape: {}".format(y_train.shape)) >> y_train shape: (79,)
# print("X_test shape: {}".format(X_test.shape)) >> X_test shape: (20, 42)
# print("y_test shape: {}".format(y_test.shape)) >> y_test shape: (20,)

# create a random forest classifier
model = RandomForestClassifier()

# fit the model to the training set 
model.fit(X_train, y_train)

# predict the labels for the test set
y_predict = model.predict(X_test)

# calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100)) 

# save the model in a pickle file 
f = open('model.pickle', 'wb')

# dump the model as a key value pair in the pickle file 
pickle.dump({'model': model}, f)
f.close()


