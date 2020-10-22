import time
import os
import json
from collections import OrderedDict
import flask
from flask import Flask
from flask import request
from read import *
import numpy as np
import trie as trie
import bigram as bigram
from redis import StrictRedis, ConnectionPool
import pymysql.cursors
pool = ConnectionPool(host='kika-backend-test1.intranet.com', port=6379, db=2, password='',decode_responses=True)
redis = StrictRedis(connection_pool=pool)
connection = pymysql.connect(host='172.31.25.236', port=3306, user='rwuser', passwd='xinmei365',db='kika_online',charset='utf8', cursorclass=pymysql.cursors.DictCursor)
#languages = ['ar','es_US','ms_MY','ru','en_us']
languages = ['ar','en_us']
stopword={'en_us':['.',',','!','?','#','!!!','!','..','...','??',':','"','...'],'ar':['.']}
unigram_trees={}
bigram_dicts={}
lstm_models = {}
#d1=redis.
d1={}
for language in languages:
    tree = trie.build(language+"/"+language+"_unigram", is_case_sensitive=False)
    unigram_trees[language]=tree
    bigramdict_path = language+'/'+language+'_bigram'
    bi_dict = bigram.genDictN(bigramdict_path,2)
    bigram_dicts[language]=bi_dict
    #print(tree)
    #print(bi_dict)
#     model = model_for_test_lstm.LSTMModel(language+'/'+language+'.pb',language)
#     lstm_models[language] = model
# sess = tf.Session()

user_state = {}

app = Flask(__name__)

# def getState(userid):
#     if userid not in user_state:
#         lm_output_state = np.zeros([2, 2, 1, 400], dtype=np.float32)
#         kc_output_state = np.zeros([2, 2, 1, 400], dtype=np.float32)
#         print(lm_output_state,kc_output_state)
#         user_state[userid] = (lm_output_state,kc_output_state)
#         return (lm_output_state,kc_output_state)
#     else:
#         return user_state[userid]

# def updateState(user_id,lm_output_state,kc_output_state):
#     user_state[user_id] = (lm_output_state,kc_output_state)
#     return
# def getState_redis(userid,language):
#     l={}
#     k={}
#     id_lan_lm=userid+language+'lm'
#     id_lan_kc=userid+language+'kc'
#     if redis.exists(id_lan_lm)!=1 and redis.exists(id_lan_kc)!=1:
#     #if userid not in user_state:
#         lm_output_state = np.zeros([2, 2, 1, 400], dtype=np.float32)
#         kc_output_state = np.zeros([2, 2, 1, 400], dtype=np.float32)
#         print(lm_output_state,kc_output_state)
#         l=lm_output_state.tolist()
#         k=kc_output_state.tolist()
#         #print(l)
#         j_lm=json.dumps(l)
#         j_kc=json.dumps(k)
#         redis.set(id_lan_lm,j_lm)
#         redis.set(id_lan_kc,j_kc)
#         lm = redis.get(id_lan_lm)
#         lm = eval(lm)
#         lm_value=lm
#         lm_value=np.array(lm_value,dtype=np.float32)
#         print(lm_value.shape)
#         #for lm_value in lm.values():
#             #lm_value=np.array(lm_value)
#         #print(lm_value)
#         kc = redis.get(id_lan_kc)
#         kc = eval(kc)
#         kc_value=kc
#         kc_value=np.array(kc_value,dtype=np.float32)
#         #for kc_value in kc.values():
#             #kc_value=np.array(kc_value)
#         print(lm_value,kc_value)
#         user_state[userid]=(lm_value,kc_value)

#         return (lm_value,kc_value)
#     else:
#         lm = redis.get(id_lan_lm)
#         lm = eval(lm)
#         #for lm_value in lm.values():
#         lm_value=np.array(lm,dtype=np.float32)
#         print(lm_value)
#         kc = redis.get(id_lan_kc)
#         kc = eval(kc)
#         #for kc_value in kc.values():
#         kc_value=np.array(kc,dtype=np.float32)
#         print(kc_value)
#         user_state[userid]=(lm_value,kc_value)
#         return user_state[userid]


# def lstm_redis(model,user_id,last_word,input_letters,k,language):
#     l={}
#     a={}
#     lm_output_state,kc_output_state = getState_redis(user_id,language)
#     id_lan_lm=user_id+language+'lm'
#     id_lan_kc=user_id+language+'kc'
#     print("lm_output_state",lm_output_state.shape)
#     print("kc_output_state",kc_output_state.shape)
#     #try:
#     predict_words,predict_words_list,lm_output_state,kc_output_state = model.predict_pb(sess,last_word,input_letters,lm_output_state,kc_output_state,k)
#     print("lm_output_state",lm_output_state)
#     #except   Exception as e:print(e)
#     l=lm_output_state.tolist()
#     a=kc_output_state.tolist()
#     j_lm=json.dumps(l)
#     j_kc=json.dumps(a)
#     redis.set(id_lan_lm,j_lm)
#     redis.set(id_lan_kc,j_kc)
    
#     return predict_words_list

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

# def lstm(model,user_id,last_word,input_letters,k):
#     lm_output_state,kc_output_state = getState(user_id)
#     print(type(lm_output_state),type(kc_output_state))
#     predict_words,predict_words_list,lm_output_state,kc_output_state = model.predict_pb(sess,last_word,input_letters,lm_output_state,kc_output_state,k)
#     updateState(user_id,lm_output_state,kc_output_state)
#     return predict_words_list
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
def WriteDict2Redis(user_id,last_word,stopword,lan): 
    id_lan=user_id+lan
    #redis.set(id_lan,last_word)
    if last_word in stopword[language]:
        print("stopword")
        redis.delete(id_lan)
    else:   
        if redis.exists(id_lan)==1:
            #redis.set('last_word',last_word)
            key=redis.get(id_lan)
            print('key',key)
            keyin=last_word
            print('keyin',keyin)
            if key!=keyin:
                CreateDict(user_id,lan,key,keyin,d1)
                ##dict转为json传入到redis里
                #j = json.dumps(d1[user_id])
                #redis.set(user_id,j)
                #cachevalue = redis.get(user_id)
                #trans_value = eval(cachevalue)
                #print("trans_value",trans_value)
                redis.set(id_lan,keyin)

        else:
            print("-------")
            redis.set(id_lan,last_word)
#lstm插入到Redis
#def WriteLstm2Redis()

@app.route('/predict', methods=['GET','POST'])
def predict():
    #d1={}
    #get 语言
    language = flask.request.values.get('lan')
    tree = unigram_trees.get(language,None)
    bi_dict = bigram_dicts.get(language,None)
    #model = lstm_models.get(language,None)
    if tree is None or bi_dict is None:
        return json.dumps({'result':'not support language'}) 
    #get userid
    user_id = flask.request.values.get('user_id')
    #to redis
    redis.sadd('user_id',user_id)

    #print('user_id',user_id)
    
    last_word = flask.request.values.get('last_word')
    #print("redis",redis.get('last_word'))
    #print("redis1",redis.get('last_word'))
    WriteDict2Redis(user_id,last_word,stopword,language)
    #print(redis.exists('last_word'))
    print(d1)
    #print('last_word',last_word)
    #print('last_word')
    #connection.commit()
    input_letters = flask.request.values.get('input_letters')
    topk = int(flask.request.values.get('top_k'))
    #print(input_letters)
    if len(last_word)==0 and len(input_letters)==0:
        return json.dumps({'result':'no input'}) 
    #之前的版本
    #lstm_result = lstm(model,user_id,last_word,input_letters,topk)
    #redis的版本
    #lstm_result = lstm_redis(model,user_id,last_word,input_letters,topk,language)
    #print(lstm_result)
    result = []
    if input_letters is not '0':
        uni_result = unigram(tree,input_letters)
        result = uni_result[:topk]
    else:
        print('input_letters is 0')
        bi_result_list = bigram(bi_dict,last_word)
        #print(bi_result_list)
        result = bi_result_list[:topk]
        # for key in bi_result_list:
        #     if key not in result:
        #         result[2] = key
        #         break
    return json.dumps({'topk':result})
    # return json.dumps({'top1'})
def readSql2Redis(conn,r):
    try:
        cursor = conn.cursor()
        sql = "select user_id from user_info"
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
            sql1 = "replace into user_info (user_id) values (%s)"
            values = (i)
            cursor.execute(sql1,values)
        except   Exception :print("插入失败")

    conn.commit()

def readSql2dict(conn,d1):

    try:
        cursor = conn.cursor()
        sql = "select user_id from user_info"
        cursor.execute(sql)
        results = cursor.fetchall() 
        #print('results',results)   
        for data in results:
            #d1.setdefault(data['user_id'],{})

            sql1="select en_us from user_info where user_id=(%s)"
            values = (data['user_id'])
            cursor.execute(sql1,values)
            en_us=cursor.fetchall()
            for e in en_us:
                print(type(e['en_us']))
                if e['en_us'] is not null:
                    print('-------')
                    d_user=str(data['user_id'])
                    d1.setdefault(d_user,{})
                    d1[d_user].setdefault('en_us',eval(e['en_us']))
        print("d1",d1)

            
    except  Exception :print("查询失败")

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
                    
                    sql2 = 'update user_info set en_us ="%s" where user_id =(%s)'
                    #print(sql2)
                    #en_us=d2[j]
                    #values = (en_us)
                    #print(a)
                    cursor.execute(sql2 % (a,i))
                except   Exception as e:print("插入失败",e)

    conn.commit()


if __name__ == '__main__':
    readSql2Redis(connection,redis)
    readSql2dict(connection,d1)
    app.run(
      host='0.0.0.0',
      port= 5002,
      debug=False
    )
    try:
        pid = os.fork()

        if pid == 0:
            writeRedis2Sql(connection,redis)
            writeDict2Sql(connection,d1)
            #time.sleep(5)
            
    except   Exception :print("失败")




