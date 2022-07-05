#Importing libraries
import pickle


from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)

#Fitting model with trainig data
decision_tree.fit(X, y)

# Saving model to disk
pickle.dump(decision_tree, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 2, 5,4]]))