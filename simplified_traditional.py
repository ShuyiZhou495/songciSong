#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'tian'
__data__ = '2019/3/26 17:51'

from letter_trans import Converter #

def cat_to_chs(sentence): #传入参数为列表
        """
        将繁体转换成简体
        :param line:
        :return:
        """
        sentence =",".join(sentence)
        sentence = Converter('zh-hans').convert(sentence)
        sentence.encode('utf-8')
        return sentence.split(",")


def chs_to_cht(sentence):#传入参数为列表
        """
        将简体转换成繁体
        :param sentence:
        :return:
        """
        #sentence =",".join(sentence)
        result = ""
        punc = ['。', '，', '？', '！', '、', '“', '”']
        for word in sentence:
                if word!=' ':
                        if(word=='\n' or word in punc):
                                result = result.rstrip()
                                result += '\n'
                        else:
                                result += (word + ' ')
        result = result.rstrip()
        sentence = Converter('zh-hant').convert(result)
        sentence.encode('utf-8')
        return sentence