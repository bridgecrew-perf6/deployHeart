{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d7fe1675",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from flask import Flask, jsonify, render_template, request\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b982fa4a",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = Flask(__name__)\n",
    "model = pickle.load(open('model.pkl','rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5bafdae1",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6f0cf6c4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__' (lazy loading)\n",
      " * Environment: production\n",
      "\u001b[31m   WARNING: This is a development server. Do not use it in a production deployment.\u001b[0m\n",
      "\u001b[2m   Use a production WSGI server instead.\u001b[0m\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Restarting with stat\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\91700\\anaconda3\\envs\\test_env\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3259: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    in_features = [float(x) for x in request.form.values()]\n",
    "    final_features = [np.array(in_features)]\n",
    "    prediction = model.predict(final_features)\n",
    "    res = ''\n",
    "    if(prediction==0):\n",
    "        res = 'NO'\n",
    "    else:\n",
    "        res = 'YES'\n",
    "    prob = model.predict_proba(final_features)\n",
    "    probable_percent = prob[:1 , 1:][0][0]*100\n",
    "    \n",
    "    report = {\"prediction of heart disease : \" : res,\n",
    "              \"probaility of truth : \" : probable_percent }\n",
    "    return render_template('index.html', prediction_text = report) \n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6fa2abb9",
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
