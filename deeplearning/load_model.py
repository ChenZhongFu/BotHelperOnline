# -*- coding:utf-8 -*-
from keras.models import load_model
import cPickle as pickle
from django.db.models import Q
from models import ModelInfo
import os
import logging
import time
import numpy as np
BASE_DIR = os.path.dirname(__file__)
DIR = BASE_DIR.replace("BotHelperOnline/deeplearning","")
model_dic = {}
online_models = ModelInfo.objects.filter(~Q(online_url=''),is_online=1)

for info in online_models:
    m_key = 'classify'
    m_key += str(info.app_id)
    path =DIR+m_key+'.h5'
    model_dic[m_key] = load_model(info.online_url)
    model_dic[m_key].predict(np.zeros((1, 100)))
    if os.path.exists(path):
        os.remove(path)
    os.rename(info.online_url,path)
    info.online_url=path
    info.is_replace=0
    info.save()

with open(DIR+'word_index.pkl', 'rb') as vocab:
    model_dic['word_index'] = pickle.load(vocab)
with open(DIR+'embedding.pkl', 'rb') as vocab:
    model_dic['embedding'] = pickle.load(vocab)
def cron_load():
    replace_models = ModelInfo.objects.filter(~Q(online_url=''),is_online=1,is_replace=1)
    if not replace_models:
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+" no model replaced")
    for info in replace_models:
        m_key = 'classify'
        m_key += str(info.app_id)
        if model_dic.has_key(m_key):
            model_dic.pop(m_key)
            model_dic[m_key]=load_model(info.online_url)
        else:
            model_dic[m_key]=load_model(info.online_url)
        model_dic[m_key].predict(np.zeros((1, 100)))
        path = DIR + m_key + '.h5'
        if os.path.exists(path):
            os.remove(path)
        os.rename(info.online_url, DIR + m_key + '.h5')
        info.update(online_url=DIR + m_key + '.h5', is_replace=0)
        print("load_model    appId="+str(info.app_id)+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    if not model_dic.has_key('word_index'):
        with open(DIR + 'word_index.pkl', 'rb') as vocab:
            model_dic['word_index'] = pickle.load(vocab)
    if not model_dic.has_key('embedding'):
        with open(DIR + 'embedding.pkl', 'rb') as vocab:
            model_dic['embedding'] = pickle.load(vocab)
