from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import time
import numpy as np
import tensorflow as tf
import collections
import sys
import codecs
import re
import copy
# from model import *
class LSTMModel(object):
    # Below is the language model.
    def __init__(self,graph_file, vocab_path,past=None):
        self.start_str = "<start>"
        self.eos_str = "<eos>"
        self.unk_str = "<unk>"
        self.num_str = "<num>"
        self.pun_str = "<pun>"
        self.id2token_in_words, self.id2token_in_letters, self.id2token_out = {}, {}, {}
        self.token2id_in_words, self.token2id_in_letters, self.token2id_out = {}, {}, {}
        vocab_int_word = vocab_path + '/vocab_in_words'
        vocab_out_word = vocab_path + '/vocab_out'
        vocab_in_letters = vocab_path + '/vocab_in_letters'
        with open(vocab_int_word, mode="r", encoding="utf-8") as f:
            for line in f:
                token, id = line.split("##")
                id = int(id)
                self.id2token_in_words[id] = token
                self.token2id_in_words[token] = id
        with open(vocab_out_word, mode="r", encoding="utf-8") as f:
            for line in f:
                token, id = line.split("##")
                id = int(id)
                self.id2token_out[id] = token
                self.token2id_out[token] = id
        with open(vocab_in_letters, mode="r", encoding="utf-8") as f:
            for line in f:
                token, id = line.split("##")
                id = int(id)
                self.id2token_in_letters[id] = token
                self.token2id_in_letters[token] = id
        prefix = "import/"
        self.lm_state_in_name = prefix + "Online/WordModel/state:0"
        self.lm_input_name = prefix + "Online/WordModel/batched_input_word_ids:0"
        self.lm_state_out_name = prefix + "Online/WordModel/state_out:0"
        self.kc_top_k_name = prefix + "Online/LetterModel/top_k:0"
        self.key_length = prefix + "Online/LetterModel/batched_input_sequence_length:0"
        self.kc_state_in_name = prefix + "Online/LetterModel/state:0"
        self.kc_lm_state_in_name = prefix + "Online/LetterModel/lm_state_in:0"
        self.kc_input_name = prefix + "Online/LetterModel/batched_input_word_ids:0"
        self.kc_top_k_prediction_name = prefix + "Online/LetterModel/top_k_prediction:1"
        self.kc_output_name = prefix + "Online/LetterModel/probabilities:0"
        self.kc_state_out_name = prefix + "Online/LetterModel/state_out:0"
        with open(graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
    def predict(self, sess, features, temperature):
        return sess.run(self.logits, feed_dict={self.input_data: features})

    def letters2ids(self, letters):
        # letters_split = re.split("\\s+", letters)
        if len(letters)==0:
            return [self.token2id_in_letters[self.start_str]]
        return [self.token2id_in_letters[self.start_str]] + [self.token2id_in_letters.get(letter, 0) for letter in
                                                             letters if len(letter) > 0]
    def word2id(self, word, return_word=False, return_processed=False):
        tmp_dict = {}
        _word_processed = word
        word_out = copy.deepcopy(word)
        END_WITH_SYMBOL_REGEX_DEFAULT = re.compile("([a-zA-Z']+)([^a-zA-Z']+$)")
        SYMBOL_SERIAL_REGEX = re.compile("^[^a-z0-9A-Z']+$")
        if re.search(SYMBOL_SERIAL_REGEX, word):
            tmp_dict["id"] = self.token2id_out.get(self.unk_str)
            tmp_dict["out"] = self.unk_str
            tmp_dict["processed"] = word[0]
            return tmp_dict
        if word in self.token2id_out:
            word_out = word
        else:
            word = re.sub(END_WITH_SYMBOL_REGEX_DEFAULT, lambda x: x.group(1), word)
            _word_processed = copy.deepcopy(word)
            if word in self.token2id_out:
                word_out = word
            elif word.lower() in self.token2id_out:
                word_out = word.lower()
            else:
                word_out, word_processed = self.unk_str, word_out
        rid = self.token2id_out.get(word_out, -1)
        tmp_dict["id"] = rid
        if return_word:
            tmp_dict["out"] = word_out
        if rid == -1:
            tmp_dict["id"] = self.token2id_out[self.unk_str]
        tmp_dict["processed"] = _word_processed
        return tmp_dict
    def words2ids(self, words):
        #return [self.eos_id] + [self.word2id(word) for word in words if len(word) > 0]
        return [self.word2id(word) for word in words if len(word) > 0]
    def sentence2list(self, sentence):
        words_array = re.split('\\s+', sentence)
        # word_letters = words_array[-1]
        new_words_array = [self.eos_str] + words_array[:-1]
        labels_array = words_array
        # letters = ' '.join(word_letters)
        # words_ids = self.words2ids(words_array)
        # letters_ids = self.letters2ids(letters)
        return new_words_array, labels_array
    def predict_pb(self,sess,last_word,input_letters,lm_state_out,k):
        #print("kc_output_state",type(kc_state_out))
        letters_ids = self.letters2ids(input_letters)  # 输出单词字母转id
        print(letters_ids)
        # prediction_made+=1
        input_letter_idx = self.word2id(self.start_str)['id']
        # if last_word is not None:  # 去掉起始位
        #     input_x = self.word2id(last_word)['id']
            #input_x = np.array([[input_x]], dtype=np.int32)
        out_str_list = []
        input_str_list = []
        probability_topk_list = []

        # lm_state_out = np.zeros([2, 2, 1, 400], dtype=np.float32)
        # kc_state_out = np.zeros([2, 2, 1, 400], dtype=np.float32)

        words_out = []
        probs_out = []
        kc_state_out=None
        #使用语言模型，此时
        #当前输入为第一个单词，没输入完
        # if len(last_word)==0:
        #     feed_values = {self.lm_input_name: [[input_x]]}
        #     lm_state_out = sess.run([self.lm_state_out_name], feed_dict=feed_values)[0]
        #elif len(last_word)>0 and len(input_letters) == 0:
        feed_values = {self.kc_top_k_name: k, self.key_length: [1]}
        if len(last_word) > 0:
            #假如前面有词，现在需要输入2个，一个是前面的词，一个是之前的LM_STATE
            # Phase I: read contexts.
            input_x = self.word2id(last_word)['id']
            feed_value = {self.lm_input_name: [[input_x]]}
            feed_value[self.lm_state_in_name] = lm_state_out
            lm_state_out = sess.run([self.lm_state_out_name], feed_dict=feed_value)[0]
            feed_values[self.kc_lm_state_in_name] = lm_state_out
            feed_values[self.kc_input_name]=[[input_letter_idx]]
            probabilities, top_k_predictions = sess.run(
                [self.kc_output_name, self.kc_top_k_prediction_name],
                feed_dict=feed_values)
        # Phase II: read letters, predict by feed the letters one-by-one.
        #如果input lette等于0
        else:
            for i in range(len(letters_ids)):
                feed_values[self.kc_input_name]=[[letters_ids[i]]]
                if i==0:
                    feed_values[self.kc_state_in_name] = lm_state_out
                else:
                    feed_values[self.kc_state_in_name] = kc_state_out
            # Use letter model's final state to letter model's initial state when feed the letters one-by-one.
                probabilities, top_k_predictions, kc_state_out = sess.run([self.kc_output_name, self.kc_top_k_prediction_name,
                                                                                 self.kc_state_out_name], feed_dict=feed_values)
        predict_words = {}
        predict_words_list=[]
        i = 0
        stop_words=['<unk>','<und>','<eos>']
        for idx in top_k_predictions[0]:
            print('idx :',idx)
            key=self.id2token_out.get(idx,'<unk>')
            if key not in stop_words:
                predict_words[key] = int(probabilities[0][i]*1000)
                predict_words_list.append(key)
                i+=1
        return predict_words,predict_words_list,lm_state_out,kc_state_out

    def testlite_lstm(self, sess,last_word,input_letters,lm_output_state,kc_output_state):
        letters_ids = self.letters2ids(input_letters)  # 输出单词字母转id
        # prediction_made+=1
        if last_word is not None:  # 去掉起始位
            input_x = self.word2id(last_word)['id']
            print(input_x)
            input_x = np.array([input_x], dtype=np.int32)

            self.lm_interpreter.set_tensor(self.lm_input_details[0]['index'], input_x)
            self.lm_interpreter.set_tensor(self.lm_input_details[1]['index'], lm_output_state)

            self.lm_interpreter.invoke()
            print('invoke!!!')
            lm_output_state = self.lm_interpreter.get_tensor(self.lm_output_details[0]['index'])

        for j in range(len(letters_ids)):
            input_letter = letters_ids[j]
            input_letter = np.array([input_letter], dtype=np.int32)
            if j == 0 and last_word is not None:
                kc_input_state = lm_output_state
            else:
                kc_input_state = kc_output_state
            self.kc_interpreter.set_tensor(self.kc_input_details[0]['index'], input_letter)
            self.kc_interpreter.set_tensor(self.kc_input_details[1]['index'], kc_input_state)
            self.kc_interpreter.set_tensor(self.kc_input_details[2]['index'], np.array(10, dtype=np.int32))
            # interpreter.set_tensor(input_details[1]['index'], np.array(30, dtype=np.int32))
            self.kc_interpreter.invoke()
            print('invoke!!!')
            kc_output_state = self.kc_interpreter.get_tensor(self.kc_output_details[0]['index'])
            kc_output_prob = self.kc_interpreter.get_tensor(self.kc_output_details[1]['index'])
            kc_output_idx = self.kc_interpreter.get_tensor(self.kc_output_details[2]['index'])
            print(kc_output_idx)
            # print(kc_output_prob)
        predict_words = {}
        predict_words_list=[]
        i = 0
        stop_words=['<unk>','<und>','<eos>']
        for idx in kc_output_idx[0]:
            key=self.id2token_out.get(idx,'<unk>')
            if key not in stop_words:
                predict_words[key] = int(kc_output_prob[0][i]*1000)
                predict_words_list.append(key)
            i+=1
        return predict_words,predict_words_list,lm_output_state,kc_output_state
