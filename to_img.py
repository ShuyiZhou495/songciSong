import uuid
from music21 import *

def get_img(json_input):
    """
    :param json_input:
    :return the path of generated 五线谱: eg, 'static/upload/img/output_img1.jpg':
    """
    
    #把json_input变成列表
    lyric=[]
    duration=[]
    note=[]
    for i in range(len(json_input)):
        lyric.append(json_input[i]["lyric"]+' &')
        duration.append(json_input[i]["duration"]+[1])
        note.append(json_input[i]["key"]+['rest'])
    lyric=[i for x in lyric for i in x.split(' ')]
    duration=[i for x in duration for i in x]
    note=[i for x in note for i in x]
    
    for i in range(len(lyric)):
        if lyric[i]=='&':
            lyric[i]=' '

    #把一个字的多个音分开
    for i in range(len(note)):
        if len(note[i])>=5:
            
            if len(note[i])==5:
                lyric[i]=[lyric[i],' ']
                duration=[duration[i]/2,duration[i]/2]
            if len(note[i])==8:
                lyric[i]=[lyric[i],' ',' ']
                duration=[duration[i]/3,duration[i]/3,duration[i]/3]
            if len(note[i])==11:
                lyric[i]=[lyric[i],' ',' ',' ']
                duration=[duration[i]/4,duration[i]/4,duration[i]/4,duration[i]/4]
            note[i]=note[i].split(',')
    lyric=[i for x in lyric for i in x]
    duration=[i for x in duration for i in x]
    note=[i for x in note for i in x]
        
    #调整音的长度
    duration=['0.25' if x=='0.2' else x for x in duration]
    duration=['0.75' if x=='0.7' else x for x in duration]
    duration=['1.25' if x=='1.2' else x for x in duration]
    
    #把note转化为标准音高
    temp=[]
    pit=['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
    for i in range(len(note)):
        if note[i] == 'rest':
            temp.append('rest')
        else:
            num = (int(note[i])-67)%12
            if num in range(12):
                temp.append(pit[num])

    
    for i in range(len(note)):
        if temp[i] != 'rest':
            note[i]=temp[i] + str(4+(int(note[i])-67)//12)
        
        
    #连接乐谱
    m= stream.Stream()
    m.append(meter.TimeSignature('4/4'))
    for i in range(len(note)):
        if note[i] == 'rest':
            temp=note.Rest()
            temp.duration=duration.Duration(1)
            temp.lyric=''
        else:
            temp=note.Note(note[i])
            temp.duration=duration.Duration(float(duration[i]))
            temp.lyric=lyric[i]
        m.append(temp)
    

    us = environment.UserSettings()
    us['lilypondPath'] = '~/bin/lilypond'
    conv =  music21.converter.subConverters.ConverterLilypond()
    uuid_str = uuid.uuid4().hex
    temp_file_name = 'tmpfile_%s' % uuid_str
    filePath = 'static/upload/img/'+temp_file_name
    conv.write(m, fmt = 'lilypond', fp=filePath, subformats = ['png'])
    
    return filePath+'.png'

if __name__ == '__main__':
    s = [{'lyrics': '還 愛 著 你 你 卻 把 別 人 擁 在 懷 裡', 'key': ['77', '77', '77', '80', '77', '77', '77', '77', '77', '76', '74', '72', '74'],'duration': ['0.2', '0.2', '0.2', '0.4', '0.7', '0.7', '1.3', '0.2', '0.2', '0.2', '0.2', '0.2', '0.4']}, {'lyrics': '不 能 再 這 樣 下 去', 'key': ['67', '74', '72', '69', '69', '67', '72'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '穿 過 愛 的 暴 風 雨', 'key': ['68', '63', '63', '65,68', '67', '72', '63'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '寧 願 清 醒 忍 痛 地 放 棄 你', 'key': ['65', '67', '69', '69', '67', '67', '65', '65', '67', '65'],'duration': ['0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3']}, {'lyrics': '還 愛 著 你 你 卻 把 別 人 擁 在 懷 裡', 'key': ['77', '77', '77', '80', '77', '77', '77', '77', '77', '76', '74', '72', '74'],'duration': ['0.2', '0.2', '0.2', '0.4', '0.7', '0.7', '1.3', '0.2', '0.2', '0.2', '0.2', '0.2', '0.4']}, {'lyrics': '不 能 再 這 樣 下 去', 'key': ['67', '74', '72', '69', '69', '67', '72'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '穿 過 愛 的 暴 風 雨', 'key': ['68', '63', '63', '65,68', '67', '72', '63'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '寧 願 清 醒 忍 痛 地 放 棄 你', 'key': ['65', '67', '69', '69', '67', '67', '65', '65', '67', '65'],'duration': ['0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3']}, {'lyrics': '還 愛 著 你 你 卻 把 別 人 擁 在 懷 裡', 'key': ['77', '77', '77', '80', '77', '77', '77', '77', '77', '76', '74', '72', '74'],'duration': ['0.2', '0.2', '0.2', '0.4', '0.7', '0.7', '1.3', '0.2', '0.2', '0.2', '0.2', '0.2', '0.4']}, {'lyrics': '不 能 再 這 樣 下 去', 'key': ['67', '74', '72', '69', '69', '67', '72'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '穿 過 愛 的 暴 風 雨', 'key': ['68', '63', '63', '65,68', '67', '72', '63'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '寧 願 清 醒 忍 痛 地 放 棄 你', 'key': ['65', '67', '69', '69', '67', '67', '65', '65', '67', '65'],'duration': ['0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3']}, {'lyrics': '還 愛 著 你 你 卻 把 別 人 擁 在 懷 裡', 'key': ['77', '77', '77', '80', '77', '77', '77', '77', '77', '76', '74', '72', '74'],'duration': ['0.2', '0.2', '0.2', '0.4', '0.7', '0.7', '1.3', '0.2', '0.2', '0.2', '0.2', '0.2', '0.4']}, {'lyrics': '不 能 再 這 樣 下 去', 'key': ['67', '74', '72', '69', '69', '67', '72'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '穿 過 愛 的 暴 風 雨', 'key': ['68', '63', '63', '65,68', '67', '72', '63'],'duration': ['0.2', '0.2', '0.2', '0.2', '0.2', '0.2', '0.2']},{'lyrics': '寧 願 清 醒 忍 痛 地 放 棄 你', 'key': ['65', '67', '69', '69', '67', '67', '65', '65', '67', '65'],'duration': ['0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3', '0.3']}]
    get_img(s)

