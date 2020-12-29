import time
import os
import model_for_test_lstm
import tensorflow as tf
import json
import flask
from flask import Flask
import logging
from online_logging import config_logging
from collections import OrderedDict
from flask import request
from read import *
import numpy as np
import trie as trie
import bigram as bigram
#from sshtunnel import SSHTunnelForwarder
#languages = ['ar','es_US','ms_MY','ru','en_us']
languages = ['en_us','es_us','ms_my','ar','ru']
stopword={'en_us':['.',',','!','?','#','!!!','!','..','...','??',':','"','...'],'ar':['.'],'ru':['.'],'es_us':['.'],'ms_my':['.']}
unigram_trees={}
bigram_dicts={}
lstm_models = {}
#d1=redis.
d1={}
config_logging(logging,'do_test_lstm.log')
for language in languages:
    #tree = trie.build(language+"/"+language+"_unigram", is_case_sensitive=False)
    #unigram_trees[language]=tree
    #bigramdict_path = language+'/'+language+'_bigram'
   #bi_dict = bigram.genDictN(bigramdict_path,2)
    #bigram_dicts[language]=bi_dict
    #print(tree)
    #print(bi_dict)
    model = model_for_test_lstm.LSTMModel(language+'/'+language+'.pb',language)
    lstm_models[language] = model
sess = tf.Session()
user_state = {}
app = Flask(__name__)
def getState(userid):
    if userid not in user_state:
        lm_output_state = np.zeros([2, 2, 1, 400], dtype=np.float32)
        #print(lm_output_state,kc_output_state)
        user_state[userid] = lm_output_state
        return lm_output_state
    else:
        return user_state[userid]

def updateState(user_id,lm_output_state,kc_output_state):
    user_state[user_id] = lm_output_state
    return
def unigram(tree,input_letters):
    uni_result = {}
    uni_result_list = []
    for key, node in trie.search(tree, input_letters, limit=10):
        uni_result[key]=node.weight
        uni_result_list.append(key)
    return uni_result_list

def bigram(bi_dict,last_word):
    bi_result=bi_dict[last_word.lower()]
    bi_result_list = []
    for key in bi_result:
        bi_result_list.append(key)
    return bi_result_list

def lstm(model,user_id,last_word,input_letters,k):
    lm_output_state = getState(user_id)
    #print(type(lm_output_state),type(kc_output_state))
    predict_words,predict_words_list,lm_output_state,kc_output_state = model.predict_pb(sess,last_word,input_letters,lm_output_state,k)
    updateState(user_id,lm_output_state,kc_output_state)
    return predict_words_list

def CreateDict(user_id,lan,key,keyin,d1):
    #d={}
    if user_id in d1.keys():
        if lan in d1[user_id].keys():
            if key in d1[user_id][lan].keys():
                if keyin in d1[user_id][lan][key].keys():
                    d1[user_id][lan][key][keyin]=d1[user_id][lan][key][keyin]+1
                else:
                    d1[user_id][lan].setdefault(key,{})[keyin]=1
            else:
                d1[user_id][lan].setdefault(key,{})[keyin]=1
        else:
            d1[user_id].setdefault(lan,{})
            d1[user_id][lan].setdefault(key,{})[keyin]=1
    else:
        d1.setdefault(user_id,{})
        d1[user_id].setdefault(lan,{})
        d1[user_id][lan].setdefault(key,{})[keyin]=1
    return d1

#lstm插入到Redis
#def WriteLstm2Redis()
def error_detection(lan,user_id,last_word,input_letters,top_k,AuthorizeToken):
    '''
    判断错误类型
    '''
    if len(lan)==0:
        return json.dumps({'errMessage':'field \"lan\" must not empty',"status": "fail", "errCode": "-1"})
    if len(user_id)==0:
        return json.dumps({'errMessage':'field \"user_id\" must not empty',"status": "fail", "errCode": "-1"})
    if len(last_word)!=0 and len(input_letters)!=0:
        return json.dumps({'errMessage':'field \"last_word\" and \"input_letters\" have and only one is not empty',"status": "fail", "errCode": "-1"})
    if len(last_word)==0 and len(input_letters)==0:
        return json.dumps({'errMessage':'field \"last_word\" and \"input_letters\" have and only one is not empty',"status": "fail", "errCode": "-1"})
    if len(top_k)==0:
        return json.dumps({'errMessage':'field \"top_k\" must not empty',"status": "fail", "errCode": "-1"})
    if AuthorizeToken!='asdfzxcv':
        return json.dumps({"status": "fail", "errCode": "-1", "errMessage": "Invalid Authorize Token"})
    return None

@app.route('/predict', methods=['GET','POST'])
def predict():
    '''
    目前仅仅使用lstm进行预测
    '''
    AuthorizeToken=flask.request.values.get('AuthorizeToken')
    language = flask.request.values.get('lan')
    user_id = flask.request.values.get('user_id')
    last_word = flask.request.values.get('last_word')
    input_letters = flask.request.values.get('input_letters')
    topk = flask.request.values.get('top_k')
    logging.info(' user_id:%s last_word%s input_letters:%s'%(user_id,last_word,input_letters))
    error_message=error_detection(lan=languages,user_id=user_id,last_word=last_word,input_letters=input_letters,
                                  top_k=topk,AuthorizeToken=AuthorizeToken)
    topk=int(topk)
    if error_message is not None:
        return error_message
    try:
        model = lstm_models.get(language, None)
        lstm_result = lstm(model,user_id,last_word,input_letters,topk)
        return json.dumps({"status":"success","candidates":lstm_result})
    except:
        return json.dumps({"status":"fail","errCode":-2,"errMessage":"inference error"})
    # return json.dumps({'top1'})
def readSql2Redis(conn,r):
    try:
        cursor = conn.cursor()
        sql = "select user_id from user_info_rnn"
        cursor.execute(sql)
        results = cursor.fetchall() 
        print('results',results)   
        for data in results:
            r.sadd('user_id',data['user_id'])
    except  Exception :print("查询失败")
def writeRedis2Sql(conn,r):
    x={}
    x=r.smembers('user_id')
    cursor=conn.cursor()
    for i in x:
        try:
            sql1 = "replace into user_info_rnn (user_id) values (%s)"
            values = (i)
            cursor.execute(sql1,values)
        except   Exception :print("插入失败")
    conn.commit()
def readSql2dict(conn,d1):
    #try:
    cursor = conn.cursor()
    sql = "select user_id from user_info_rnn"
    cursor.execute(sql)
    results = cursor.fetchall() 
    print('results',results)   
    for data in results:
        #d1.setdefault(data['user_id'],{})

        sql1="select en_us from user_info_rnn where user_id=(%s)"
        values = (data['user_id'])
        cursor.execute(sql1,values)
        en_us=cursor.fetchall()
        for e in en_us:
            #print(type(e['en_us']))
            if e['en_us'] is not None:
                d_user=str(data['user_id'])
                d1.setdefault(d_user,{})
                d1[d_user].setdefault('en_us',eval(e['en_us']))

        sql2="select ar from user_info_rnn where user_id=(%s)"
        values=(data['user_id'])
        cursor.execute(sql2,values)
        ar=cursor.fetchall()
        for e in ar:
            if e['ar'] is not None:
                d_user=str(data['user_id'])
                d1.setdefault(d_user,{})
                d1[d_user].setdefault('ar',eval(e['ar']))

        sql3="select es_us from user_info_rnn where user_id=(%s)"
        values=(data['user_id'])
        cursor.execute(sql3,values)
        es_us=cursor.fetchall()
        for e in es_us:
            if e['es_us'] is not None:
                d_user=str(data['user_id'])
                d1.setdefault(d_user,{})
                d1[d_user].setdefault('es_us',eval(e['es_us']))

        sql4="select ms_my from user_info_rnn where user_id=(%s)"
        values=(data['user_id'])
        cursor.execute(sql4,values)
        ms_my=cursor.fetchall()
        for e in ms_my:
            if e['ms_my'] is not None:
                d_user=str(data['user_id'])
                d1.setdefault(d_user,{})
                d1[d_user].setdefault('ms_my',eval(e['ms_my']))

        sql5="select ru from user_info_rnn where user_id=(%s)"
        values=(data['user_id'])
        cursor.execute(sql5,values)
        ru=cursor.fetchall()
        for e in ru:
            if e['ru'] is not None:
                d_user=str(data['user_id'])
                d1.setdefault(d_user,{})
                d1[d_user].setdefault('ru',eval(e['ru']))

    print("d1",d1)
    #except  Exception :print("查询失败")
#数据库的问题{}输入不进去
def writeDict2Sql(conn,d1):
    cursor=conn.cursor()
    x=d1.keys()
    for i in x :
        y=d1[i].keys()
        d2=d1[i]
        for j in y:
            if j=='en_us':
                try:
                    #user_dict = json.loads(user_info)
                    #en_us=d2[j]
                    info_dict = d2[j]
                    # dumps 将数据转换成字符串
                    #info_json = json.dumps(info_dict,sort_keys=False,separators=(',', ': '))
                    # 显示数据类型
                    a=str(info_dict)
                    
                    sql2 = 'update user_info_rnn set en_us ="%s" where user_id =(%s)'
                    #print(sql2)
                    #en_us=d2[j]
                    #values = (en_us)
                    #print(a)
                    cursor.execute(sql2 % (a,i))
                except   Exception as e:print("插入失败",e)
            elif j=='ar':
                try:
                    info_dict=d2[j]
                    a=str(info_dict)
                    sql2='update user_info_rnn set ar="%s" where user_id=(%s)'
                    cursor.execute(sql2%(a,i))
                except Exception as e:print("插入失败",e)
            elif j=='es_us':
                try:
                    info_dict=d2[j]
                    a=str(info_dict)
                    sql2='update user_info_rnn set es_us="%s" where user_id=(%s)'
                    cursor.execute(sql2%(a,i))
                except Exception as e:print("插入失败",e)
            elif j=='ms_my':
                try:
                    info_dict=d2[j]
                    a=str(info_dict)
                    sql2='update user_info_rnn set ms_my="%s" where user_id=(%s)'
                    cursor.execute(sql2%(a,i))
                except Exception as e:print("插入失败",e)
            elif j=='ru':
                try:
                    info_dict=d2[j]
                    a=str(info_dict)
                    sql2='update user_info_rnn set ru="%s" where user_id=(%s)'
                    cursor.execute(sql2%(a,i))
                except Exception as e:print("插入失败",e)

    conn.commit()
if __name__ == '__main__':
    app.run(
      host='0.0.0.0',
      port= 5003,
      debug=False
    )





