# -*- coding:utf-8 -*-
from django.shortcuts import render
from models import ModelInfo
from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from django.db.models import Q
import jieba
import sys
from django.http import JsonResponse
import cPickle as pickle
from load_model import model_dic
import requests
import pyemd
from numpy import zeros,double, sqrt, sum as np_sum
import os
reload(sys)
sys.setdefaultencoding('utf-8')
#参数设置
BASE_DIR = os.path.dirname(__file__)
BASE_DIR = BASE_DIR.replace("BotHelperOnline/deeplearning","")

classify_sequence_length = 100 #分类模型sequence最大长度
match_sequence_length = 12 #匹配模型sequence最大长度
save_list = [u'ios', u'app', u'pk', u'tts']
#question_class = ['活动专区','客户端下载及使用','课外乐园','学习中心','账号']

#对问题分词并处理非中文字符
def cut_unchinese(raw_string):
    result = []
    seg_list = jieba.cut(raw_string.lower())
    for seg in seg_list:
        seg = ''.join(seg.split())
        seg = seg.decode(encoding='utf-8')
        seg_result = ''
        if seg in save_list:
            result.append(seg)
        else:
            if seg != u'' and seg != u"\n" and seg != u"\n\n":
                for char in seg:
                    if (char >= u'\u4e00') and (char <= u'\u9fa5'):
                        seg_result += char
                if len(seg_result) > 0:
                    result.append(seg_result)
    return result

#添加running model
def add_model(appId,m_type,m_key):
    m_info = ModelInfo.objects.filter(is_online=1,app_id=appId,type=m_type)
    if m_info:
        model_dic[m_key] = load_model(m_info[0].online_url)
        return True
    else:
        return False


# 获取所有running model的字典
def get_models(request):
    result = {'retCode': '0',
              'models':model_dic.keys(),
              'retDesc': '获取running models成功'}
    return JsonResponse(result)


#预测用户问题，返回5个最相似标准问题
def predict(request):
    #接收参数

    app_id = request.GET.get('appId','-1')
    user_question = request.GET.get('userQuestion','-1')
    use_class_model = request.GET.get('useClass','-1')
    #use_class_model=0
    if user_question == '-1' or app_id == '-1' or use_class_model =='-1':
        return JsonResponse({'retCode':'1000','retDesc':'参数错误'})

    #获取model的key
    online_model = ModelInfo.objects.filter(app_id=app_id,is_online=1)
    if not online_model:
        return JsonResponse({'retCode': '1001', 'retDesc': '没有该模型'})

    
    # model_replace1 = ModelInfo.objects.filter(app_id=app_id, is_online=1, is_replace=1)
    # x=model_replace1[0].online_url
    # if model_replace1:
    #       model_dic.pop('classify'+str(app_id))
    #       model_dic['classify' + str(app_id)] = load_model(model_replace1[0].online_url)
    #       model_replace1[0].update(is_replace=0)
    # match_model_key = 'match' + str(app_id)
    # if not model_dic.has_key(match_model_key):
    #     if not add_model(app_id,1,match_model_key):
    #         return JsonResponse({'retCode': '1001', 'retDesc': '没有该模型'})
    if not model_dic.has_key('word_index'):
        with open(BASE_DIR+'word_index.pkl', 'rb') as vocab:
            model_dic['word_index'] = pickle.load(vocab)

    user_question_vec = [] #index后的用户问题

    for word in cut_unchinese(user_question):
        if word in model_dic['word_index']:
            user_question_vec.append(model_dic['word_index'][word])
        else:
            user_question_vec.append(0)
    #########
    # 计算分类
    #########
    classify_sequences=None
    classify_predict=None
    classify_predict_argsort=None
    stand_data = []
    stand_questions = []
    class_flag=0

    if model_dic.has_key('classify'+str(app_id)) and int(use_class_model):
        class_flag = 1
        classify_sequences = [user_question_vec]
        #padding the sequences to the same length
        classify_sequences = pad_sequences(classify_sequences, maxlen = classify_sequence_length)
        classify_predict = model_dic['classify' + str(app_id)].predict(classify_sequences)
        #降序排序
        classify_predict_argsort = np.argsort(-classify_predict)
        for i in xrange(0, 2):
            para = {'tid': classify_predict_argsort[0, i],'appId':app_id,'useClass':1}
            get_r = requests.get('http://182.92.4.200:3210/AIaspect/predict', params=para)
            if get_r.status_code != requests.codes.ok:
                return JsonResponse({'retCode': '1002', 'retDesc': '知识库请求失败'})
            data = get_r.json()
            if data['userQuestion']:
                stand_data.extend(data['userQuestion'])

        for i in xrange(len(stand_data)):
            # lmb修改了此处
            stand_questions.append(stand_data[i]['question'])
    else:
        para = {'tid':-1,'appId': app_id,'useClass':0}
        get_r = requests.get('http://182.92.4.200:3210/AIaspect/predict', params=para)
        if get_r.status_code != requests.codes.ok:
            return JsonResponse({'retCode': '1002', 'retDesc': '知识库请求失败'})
        data = get_r.json()
        if data['reqResult']:
            stand_data.extend(data['reqResult'])

        for item in stand_data:
            stand_questions.append(item['question'])
    ###########
    # 计算匹配值
    ###########
    # 知识库返回的问题列表
    #stand_questions_vec = []  #index后的问题列表


    # for i in xrange(len(stand_questions)):
    #     q_vec = []
    #    for word in cut_unchinese(stand_questions[i]):
    #         if word in model_dic['word_index']:
    #             q_vec.append(model_dic['word_index'][word])
    #         else:
    #             q_vec.append(0)
    #     stand_questions_vec.append(q_vec)
    #
    # match_stand_sequences = pad_sequences(stand_questions_vec, maxlen=match_sequence_length)
    # user_question_vec = []  # index后的用户问题
    # for word in cut_unchinese(user_question):
    #     if word in word_index:
    #         user_question_vec.append(word_index[word])
    # match_user_sequence = [user_question_vec]
    # match_user_sequences = np.repeat(match_user_sequence,len(stand_questions),axis=0)
    # match_user_sequences = pad_sequences(match_user_sequences, maxlen=match_sequence_length)
    #
    # match_predict = model_dic[match_model_key].predict([match_user_sequences,match_stand_sequences])
    # match_predict_argsort = np.argsort(-match_predict,axis=0)

    # 计算用户问题与知识库问题列表的问题的距离
    dis = np.zeros(len(stand_data))
    for i in xrange(len(stand_data)):
        dis[i] = two_sentence_dis(cut_unchinese(user_question),cut_unchinese(stand_questions[i]))
    dis_argsort = dis.argsort()

    # 要返回的5个最相似问题
    q5_id = []
    q5_str = []
    class_str = None
    if class_flag:
        class_str = '%d,%d'%(classify_predict_argsort[0, 0], classify_predict_argsort[0, 1])
    for i in xrange(5):
        q5_id.append(stand_data[dis_argsort[i]]['qid'])
        # lmb修改了此处
        q5_str.append(stand_data[dis_argsort[i]]['question'])
    result = {'retCode': '0',
              'questionIdList': q5_id,
              'questionList':q5_str,
              'questionType': class_str,
              'retDesc': '获取相似前5问题成功'}
    return JsonResponse(result)

#计算两个句子的距离
def two_sentence_dis(sentence1, sentence2):
    if not model_dic.has_key('embedding'):
        with open(BASE_DIR+'embedding.pkl', 'rb') as vocab:
            model_dic['embedding'] = pickle.load(vocab)

    len_sentence1 = len(sentence1)
    len_sentence2 = len(sentence2)

    # Remove out-of-vocabulary words.
    sentence1 = [model_dic['word_index'].get(token) for token in sentence1 if model_dic['word_index'].has_key(token)]
    sentence2 = [model_dic['word_index'].get(token) for token in sentence2 if model_dic['word_index'].has_key(token)]

    diff1 = len_sentence1 - len(sentence1)
    diff2 = len_sentence2 - len(sentence2)
    if diff1 > 0 or diff2 > 0:
        print ('Removed %d and %d OOV words from document 1 and 2 (respectively).',
               diff1, diff2)

    if len(sentence1) == 0 or len(sentence2) == 0:
        print ('At least one of the documents had no words that were'
               'in the vocabulary. Aborting (returning inf).')
        return float('inf')

    dictionary_temp = list(set(sentence1 + sentence2))
    dictionary = dict(enumerate(dictionary_temp))
    vocab_len = len(dictionary)

    sen_set1 = set(sentence1)
    sen_set2 = set(sentence2)

    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if not t1 in sen_set1 or not t2 in sen_set2:
                continue
            # 计算距离
            distance_matrix[i, j] = sqrt(np_sum((model_dic['embedding'][t1] - model_dic['embedding'][t2]) ** 2))

    if np_sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        print ('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def doc2bow(document, dictionary):
        freq_dic = dict()
        for i in document:
            if freq_dic.has_key(i):
                freq_dic[i] = freq_dic[i] + 1
            else:
                freq_dic[i] = 1

        return_freq = dict()
        for i in range(len(document)):
            if return_freq.has_key(i):
                for key in range(len(dictionary)):
                    if dictionary[key] == document[i]:
                        return_freq[key] = freq_dic[document[i]]
            else:
                for key in range(len(dictionary)):
                    if dictionary[key] == document[i]:
                        return_freq[key] = freq_dic[document[i]]
        return return_freq

    def nbow(document):
        d = zeros(vocab_len, dtype=double)
        nbow = doc2bow(document, dictionary)  # Word frequencies.
        doc_len = len(document)
        for (idx, freq) in nbow.items():
        #for idx, freq in nbow:
            d[idx] = float(freq) / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(sentence1)
    d2 = nbow(sentence2)

    # Compute WMD.
    #print pyemd.emd(d1, d2, distance_matrix)
    return pyemd.emd(d1, d2, distance_matrix)

def delete_model(request):
    appId=request.GET.get('appId','-1')
    if appId == '-1':
        return JsonResponse({'retCode': '1000', 'retDesc': '参数错误'})
    ModelInfo.objects.filter(app_id=appId).delete()
    if model_dic.has_key('classify'+str(appId)):
        model_dic.pop('classify'+str(appId))

        return JsonResponse({'retCode': '0', 'retDesc': '清除模型成功'})
    else:
        return JsonResponse({'retCode':'1001','retDesc':'不存在模型'})
