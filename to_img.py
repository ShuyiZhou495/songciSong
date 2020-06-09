import uuid
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
            
            if len(myNote[i])==5:
                lyric[i]=[lyric[i],' ']
                myDuration[i]=[str(float(myDuration[i])/2),str(float(myDuration[i])/2)]
            if len(myNote[i])==8:
                lyric[i]=[lyric[i],' ',' ']
                myDuration[i]=[str(float(myDuration[i])/3),str(float(myDuration[i])/3),str(float(myDuration[i])/3)]
            if len(myNote[i])==11:
                lyric[i]=[lyric[i],' ',' ',' ']
                myDuration[i]=[str(float(myDuration[i])/4),str(float(myDuration[i])/4),str(float(myDuration[i])/4),str(float(myDuration[i])/4)]
            myNote[i]=myNote[i].split(',')
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
    us['lilypondPath'] = '~/bin/lilypond'
    conv =  music21.converter.subConverters.ConverterLilypond()
    uuid_str = uuid.uuid4().hex
    temp_file_name = 'tmpfile_%s' % uuid_str
    filePath = 'static/upload/img/'+temp_file_name
    out_filepath = conv.write(m, fmt = 'lilypond', fp=filePath, subformats = ['png'])
    
    return out_filepath
