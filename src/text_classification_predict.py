#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from src.model.svm_model import SVMModel
from src.model.naive_bayes_model import NaiveBayesModel


class TextClassificationPredict(object):
    def __init__(self) -> object:
        self.test = None

    def get_train_data(self):
        #  train data
        import json
        with open('traindata.json', encoding='utf-8') as json_data:
            traindata=json.load(json_data)
            df_train = pd.DataFrame(traindata)

        #  test train data
        test_data = []
        test = input('Ban => ')
        test_data.append({"feature": test, "target": "hoi_thoi_tiet"})
        df_test = pd.DataFrame(test_data)

        # init model naive bayes
        model = SVMModel()
        clf = model.clf.fit(df_train["feature"], df_train.target)
        predicted = clf.predict(df_test["feature"])
        print(predicted)

        #connect data_train file to answer_data file
        from pandas.io.json import json_normalize
        filepath = 'answerdata.json'
        with open(filepath, encoding='utf-8') as json_file:
            chatbot_data = json.load(json_file)
            customer = [customers for index, customers in enumerate(chatbot_data) if customers.get('target') == predicted]
            try:
                if len(customer) > 0:
                    feature = customer[0]["feature"]
                    import random
                    print(random.choice(feature))
            except:
                print("Something went wrong")


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.get_train_data()