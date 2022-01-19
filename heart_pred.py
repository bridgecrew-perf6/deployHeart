{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "48af22de",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import RandomizedSearchCV, GridSearchCV\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "4290ab8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv(\"heart-disease.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "3d665f29",
   "metadata": {},
   "outputs": [],
   "source": [
    "X=data.drop(\"target\",axis=1)\n",
    "y=data[\"target\"]\n",
    "\n",
    "X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "426778a6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8524590163934426"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.random.seed(54)\n",
    "model_clf=RandomForestClassifier(max_depth = 5, max_features = 'auto', min_samples_leaf = 3, min_samples_split = 8, n_estimators = 30)\n",
    "model_clf.fit(X_train,y_train)\n",
    "model_clf.score(X_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "ece003b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8185245901639344"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_cv = cross_val_score(model_clf, X, y, cv=5, scoring='accuracy')\n",
    "model_cv.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "5f631c85",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(model_clf, open('model.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e853a71",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
