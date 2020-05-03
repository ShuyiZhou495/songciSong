# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:18:13 2020

@author: DELL
"""
import music21
from music21 import *

def get_img(json_input):
    """
    :param json_input:
    :return the path of generated 五线谱: eg, 'static/upload/img/output_img1.jpg':
    """
    
    #把json_input变成列表
    lyric=[]
    myDuration=[]
    myNote=[]
    for i in range(len(json_input)):
        lyric.append(json_input[i]["lyrics"]+' &')
        myDuration.append(json_input[i]["duration"]+[1])
        myNote.append(json_input[i]["key"]+['rest'])
    lyric=[i for x in lyric for i in x.split(' ')]
    myDuration=[i for x in myDuration for i in x]
    myNote=[i for x in myNote for i in x]
    
    for i in range(len(lyric)):
        if lyric[i]=='&':
            lyric[i]=' '

    #把一个字的多个音分开
    for i in range(len(myNote)):
        if len(myNote[i])>=5:
            myNote[i]=myNote[i].split(',')
            if len(myNote[i])==5:
                lyric[i]=[lyric[i],' ']
                myDuration[i]=[str(float(myDuration[i])/2),str(float(myDuration[i])/2)]
            if len(myNote[i])==8:
                lyric[i]=[lyric[i],' ',' ']
                myDuration[i]=[str(float(myDuration[i])/3),str(float(myDuration[i])/3),str(float(myDuration[i])/3)]
            if len(myNote[i])==11:
                lyric[i]=[lyric[i],' ',' ',' ']
                myDuration[i]=[str(float(myDuration[i])/4),str(float(myDuration[i])/4),str(float(myDuration[i])/4),str(float(myDuration[i])/4)]
            else:
                myNote[i]=myNote[i][0:2]
    flatten = lambda x: [subitem for item in x for subitem in flatten(item)] if type(x) is list else [x]
    lyric=flatten(lyric)
    myDuration=flatten(myDuration)
    myNote=flatten(myNote)
       
    #调整音的长度
    myDuration=['0.25' if x=='0.2' else x for x in myDuration]
    myDuration=['0.75' if x=='0.7' else x for x in myDuration]
    myDuration=['1.25' if x=='1.2' else x for x in myDuration]
    
    #把myNote转化为标准音高
    temp=[]
    pit=['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
    for i in range(len(myNote)):
        if myNote[i] == 'rest':
            temp.append('rest')
        else:
            num = (int(myNote[i])-67)%12
            if num in range(12):
                temp.append(pit[num])

    
    for i in range(len(myNote)):
        if temp[i] != 'rest':
            myNote[i]=temp[i] + str(4+(int(myNote[i])-67)//12)
        
        
    #连接乐谱
    m= stream.Stream()
    m.append(meter.TimeSignature('4/4'))
    for i in range(len(myNote)):
        if myNote[i] == 'rest':
            temp=note.Rest()
            temp.myDuration=duration.Duration(1)
            temp.lyric=''
        else:
            temp=note.Note(myNote[i])
            temp.myDuration=duration.Duration(float(myDuration[i]))
            temp.lyric=lyric[i]
        m.append(temp)
    


    us = environment.UserSettings()
    us['musescoreDirectPNGPath'] = '/Applications/MuseScore 3.app/Contents/MacOS/mscore'
    from music21.converter.subConverters import ConverterMusicXML
    conv_musicxml = ConverterMusicXML()
    scorename = 'output.xml'
    filepath = 'static/upload/img/'+scorename
    out_filepath = conv_musicxml.write(m, 'musicxml', fp=filepath, subformats=['png'])
    
    return scorename + '-1.png'

if __name__ == '__main__':
    s = [{'lyrics': '還 愛 著 你 你 卻 把 別 人 擁 在 懷 裡', 'key': ['77', '77', '77', '80', '77', '77', '77', '77', '77', '76', '74', '72', '74'],'duration': ['0.2', '0.2', '0.2', '0.4', '0.7', '0.7', '1.3', '0.2', '0.2', '0.2', '0.2', '0.2', '0.4']}, {'lyrics': '不 能 再 這 樣 下 去', 'key': ['67', '74', '72', '69', '69', '67', '72'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '穿 過 愛 的 暴 風 雨', 'key': ['68', '63', '63', '65,68', '67', '72', '63'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '寧 願 清 醒 忍 痛 地 放 棄 你', 'key': ['65', '67', '69', '69', '67', '67', '65', '65', '67', '65'],'duration': ['0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3']}, {'lyrics': '還 愛 著 你 你 卻 把 別 人 擁 在 懷 裡', 'key': ['77', '77', '77', '80', '77', '77', '77', '77', '77', '76', '74', '72', '74'],'duration': ['0.2', '0.2', '0.2', '0.4', '0.7', '0.7', '1.3', '0.2', '0.2', '0.2', '0.2', '0.2', '0.4']}, {'lyrics': '不 能 再 這 樣 下 去', 'key': ['67', '74', '72', '69', '69', '67', '72'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '穿 過 愛 的 暴 風 雨', 'key': ['68', '63', '63', '65,68', '67', '72', '63'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '寧 願 清 醒 忍 痛 地 放 棄 你', 'key': ['65', '67', '69', '69', '67', '67', '65', '65', '67', '65'],'duration': ['0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3']}, {'lyrics': '還 愛 著 你 你 卻 把 別 人 擁 在 懷 裡', 'key': ['77', '77', '77', '80', '77', '77', '77', '77', '77', '76', '74', '72', '74'],'duration': ['0.2', '0.2', '0.2', '0.4', '0.7', '0.7', '1.3', '0.2', '0.2', '0.2', '0.2', '0.2', '0.4']}, {'lyrics': '不 能 再 這 樣 下 去', 'key': ['67', '74', '72', '69', '69', '67', '72'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '穿 過 愛 的 暴 風 雨', 'key': ['68', '63', '63', '65,68', '67', '72', '63'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '寧 願 清 醒 忍 痛 地 放 棄 你', 'key': ['65', '67', '69', '69', '67', '67', '65', '65', '67', '65'],'duration': ['0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3']}, {'lyrics': '還 愛 著 你 你 卻 把 別 人 擁 在 懷 裡', 'key': ['77', '77', '77', '80', '77', '77', '77', '77', '77', '76', '74', '72', '74'],'duration': ['0.2', '0.2', '0.2', '0.4', '0.7', '0.7', '1.3', '0.2', '0.2', '0.2', '0.2', '0.2', '0.4']}, {'lyrics': '不 能 再 這 樣 下 去', 'key': ['67', '74', '72', '69', '69', '67', '72'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '穿 過 愛 的 暴 風 雨', 'key': ['68', '63', '63', '65,68', '67', '72', '63'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '寧 願 清 醒 忍 痛 地 放 棄 你', 'key': ['65', '67', '69', '69', '67', '67', '65', '65', '67', '65'],'duration': ['0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3']}]
    get_img(s)