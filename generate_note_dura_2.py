import torch
from torch import nn
from xpinyin import Pinyin


# 模型的定义，包括MultiHeadAttentionLayer、PositionwiseFeedforwardLayer、
# EncoderLayer、Encoder、DecoderLayer、Decoder、Seq2Seq共7个类
class MultiHeadAttentionLayer(nn.Module):
    # “多头”注意力机制
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads  # “头”的个数
        self.head_dim = hid_dim // n_heads  # 每一“头”的输出维数
        self.fc_q = nn.Linear(hid_dim, hid_dim)  # 查询
        self.fc_k = nn.Linear(hid_dim, hid_dim)  # 键
        self.fc_v = nn.Linear(hid_dim, hid_dim)  # 值
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query=key=value为输入
        batch_size = query.shape[0]
        # query = [batch size, query len, hid dim]   key、value类似
        Q = self.fc_q(query)  # 查询向量  Q = [batch size, query len, hid dim]
        K = self.fc_k(key)  # 键向量      K = [batch size, key len, hid dim]
        V = self.fc_v(value)  # 值向量    V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]   K、V类似
        # Q[0][0][0]表示第一“头”对batch中第一个数据里的第一个单词的查询向量计算结果，它将乘以Q[0][0][i]得到对第i个单词的关注度
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, seq len, seq len]
        # energy[0][0]是第一“头”对batch中第一个数据计算出的seq len*seq len个关注度分数
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, seq len, head dim]  求和结果
        x = x.permute(0, 2, 1, 3).contiguous()  # x = [batch size, seq len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)  # x = [batch size, seq len, hid dim]
        x = self.fc_o(x)  # x = [batch size, seq len, hid dim]
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    # 前馈（feed-forward）层
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))  # x = [batch size, seq len, pf dim]
        x = self.fc_2(x)  # x = [batch size, seq len, hid dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]   src_mask = [batch size, src len]
        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        # dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))  # src = [batch size, src len, hid dim]
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))  # src = [batch size, src len, hid dim]
        return src


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,  # 输入数据集的“词汇量”
                 hid_dim,  # hid_dim是n_heads的倍数
                 n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]    src_mask = [batch size, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, src len]  用来表征单词顺序
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src = [batch size, src len, hid dim] 维度改变是因为embedding
        for layer in self.layers:
            src = layer(src, src_mask)  # src = [batch size, src len, hid dim]
        return src


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]    enc_src是Encoder的计算结果
        # trg_mask = [batch size, trg len]    src_mask = [batch size, src len]

        # self attention 自注意力
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))  # trg = [batch size, trg len, hid dim]

        # encoder attention 编码-解码注意力层
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))  # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))  # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        return trg, attention


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads,
                 pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]    src_mask = [batch size, src len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)  # pos = [batch size, trg len]
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)  # output = [batch size, trg len, output dim]
        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,
                 src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to('cpu')  # src_mask = [batch size, 1, 1, src len]

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # trg_pad_mask = [batch size, 1, trg len, 1]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device='cpu')).bool()
        # trg_sub_mask = [trg len, trg len]    torch.triu：返回矩阵 下 三角部分，其余部分为0
        trg_mask = trg_pad_mask & trg_sub_mask  # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask.to('cpu')

    def forward(self, src, trg):
        src = src.to('cpu')
        trg = trg.to('cpu')
        # src = [batch size, src len]    trg = [batch size, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)  # enc_src = [batch size, src len, hid dim]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        return output, attention


# 4个字典
dura_out_vocab_dic={'<pad>':0, '<bos>':1, '0.2':2, '0.4':3, '0.3':4, '0.5':5, '0.6':6, '0.1':7, '0.8':8, '0.7':9, '0.9':10, 
'1.0':11, '1.3':12, '1.2':13, '1.1':14, '1.5':15, '1.4':16, '1.7':17, '2.3':18, '1.8':19, '0.0':20, 
'2.0':21, '1.6':22, '3.4':23, '2.1':24, '2.2':25, '2.4':26, '1.9':27, '3.0':28, '2.6':29, '3.2':30, 
'2.5':31, '2.8':32, '2.7':33, '2.9':34, '3.6':35, '3.5':36, '3.3':37, '4.0':38, '0.25':39, '3.1':40, 
'0.75':41, '3.9':42, '03':43, '1.75':44}

dura_in_vocab_dic={'<pad>':0, '<bos>':1, 'wo':2, 'ni':3, 'de':4, 'yi':5, 'shi':6, 'bu':7, 'you':8, 'zai':9, 'ai':10, 
'xiang':11, 'xin':12, 'wei':13, 'zhi':14, 'qing':15, 'ji':16, 'hui':17, 'mei':18, 'dao':19, 'ren':20, 
'li':21, 'ye':22, 'le':23, 'qi':24, 'tian':25, 'guo':26, 'yu':27, 'zhong':28, 'yao':29, 'ke':30, 
'yuan':31, 'zhe':32, 'zhao':33, 'du':34, 'sheng':35, 'jiu':36, 'shen':37, 'wu':38, 'xi':39, 'jian':40, 
'he':41, 'ge':42, 'yan':43, 'huan':44, 'shui':45, 'jin':46, 'qu':47, 'lai':48, 'duo':49, 'zi':50, 
'wang':51, 'jing':52, 'me':53, 'ru':54, 'shou':55, 'shang':56, 'na':57, 'hao':58, 'chu':59, 'di':60, 
'yong':61, 'neng':62, 'gan':63, 'jie':64, 'ta':65, 'bian':66, 'dan':67, 'shuo':68, 'ming':69, 'hou':70, 
'xing':71, 'qian':72, 'xiao':73, 'zhen':74, 'hua':75, 'fen':76, 'zui':77, 'rang':78, 'wen':79, 'xu':80, 
'bi':81, 'dui':82, 'fu':83, 'chang':84, 'lian':85, 'xian':86, 'liu':87, 'zuo':88, 'wan':89, 'feng':90, 
'kai':91, 'bei':92, 'kan':93, 'xia':94, 'tong':95, 'fang':96, 'si':97, 'tai':98, 'mo':99, 'ran':100, 
'nan':101, 'dang':102, 'fa':103, 'jue':104, 'yang':105, 'dong':106, 'ci':107, 'cheng':108, 'nian':109, 'ku':110, 
'bie':111, 'yin':112, 'dai':113, 'tou':114, 'man':115, 'dian':116, 'zou':117, 'mian':118, 'huo':119, 'er':120, 
'lei':121, 'ban':122, 'da':123, 'xie':124, 'jiao':125, 'cong':126, 'hen':127, 'ting':128, 'zhu':129, 'bao':130, 
'yue':131, 'jia':132, 'meng':133, 'cai':134, 'gu':135, 'ju':136, 'zen':137, 'ba':138, 'deng':139, 'sui':140, 
'liang':141, 'kong':142, 'ying':143, 'jiang':144, 'que':145, 'hu':146, 'kuai':147, 'ri':148, 'quan':149, 'mi':150, 
'ling':151, 'pa':152, 'fei':153, 'gei':154, 'suan':155, 'bai':156, 'hai':157, 'men':158, 'zong':159, 'gai':160, 
'lu':161, 'ruo':162, 'la':163, 'geng':164, 'shu':165, 'an':166, 'chen':167, 'ding':168, 'gao':169, 'guang':170, 
'ceng':171, 'ma':172, 'zheng':173, 'qiu':174, 'shao':175, 'guan':176, 'chi':177, 'reng':178, 'pian':179, 'suo':180, 
'huai':181, 'fan':182, 'gong':183, 'zao':184, 'cuo':185, 'gou':186, 'duan':187, 'luo':188, 'ti':189, 'gen':190, 
're':191, 'rong':192, 'shan':193, 'yun':194, 'qin':195, 'tu':196, 'kou':197, 'qie':198, 'pei':199, 'sha':200, 
'huang':201, 'zhuan':202, 'mu':203, 'leng':204, 'xue':205, 'shuang':206, 'diao':207, 'zhan':208, 'su':209, 'chun':210, 
'hong':211, 'san':212, 'lv':213, 'lang':214, 'han':215, 'tan':216, 'pang':217, 'piao':218, 'se':219, 'tao':220, 
'bing':221, 'fou':222, 'xun':223, 'ya':224, 'chuan':225, 'bo':226, 'qiang':227, 'ping':228, 'chui':229, 'hei':230, 
'gui':231, 'tui':232, 'zhang':233, 'ou':234, 'peng':235, 'she':236, 'dou':237, 'lan':238, 'po':239, 'ning':240, 
'wai':241, 'zhui':242, 'nuan':243, 'tiao':244, 'chuang':245, 'zhuang':246, 'can':247, 'pan':248, 'gua':249, 'miao':250, 
'nei':251, 'pai':252, 'chong':253, 'lao':254, 'tang':255, 'rou':256, 'ben':257, 'chou':258, 'mang':259, 'lun':260, 
'mou':261, 'xuan':262, 'song':263, 'cang':264, 'mai':265, 'zu':266, 'kao':267, 'cha':268, 'tuo':269, 'chao':270, 
'a':271, 'pao':272, 'kuang':273, 'chan':274, 'hun':275, 'cun':276, 'nai':277, 'nv':278, 'qiao':279, 'o':280, 
'che':281, 'nao':282, 'guai':283, 'luan':284, 'zhou':285, 'liao':286, 'mao':287, 'ze':288, 'e':289, 'lie':290, 
'zan':291, 'biao':292, 'tie':293, 'lin':294, 'pu':295, 'gang':296, 'juan':297, 'nong':298, 'xiu':299, 'heng':300, 
'nuo':301, 'pi':302, 'cui':303, 'ken':304, 'xiong':305, 'kun':306, 'bang':307, 'long':308, 'nu':309, 'zhuo':310, 
'ao':311, 'cao':312, 'ruan':313, 'sa':314, 'teng':315, 'te':316, 'shun':317, 'zeng':318, 'zhun':319, 'rao':320, 
'kuo':321, 'zha':322, 'pin':323, 'shuai':324, 'ce':325, 'en':326, 'mie':327, 'min':328, 'niu':329, 'ca':330, 
'jun':331, 'bin':332, 'lia':333, 'zun':334, 'kang':335, 'niang':336, 'qiong':337, 'sang':338, 'dun':339, 'lve':340, 
'niao':341, 'tun':342, 'chai':343, 'kui':344, 'ka':345, 'lou':346, 'qun':347, 'sao':348, 'wa':349, 'zhua':350, 
'cu':351, 'pie':352, 'sai':353, 'sun':354, 'ang':355, 'diu':356, 'kua':357, 'zhai':358, 'beng':359, 'gun':360, 
'kuan':361, 'shai':362, 'tuan':363, 'za':364, 'zang':365, 'zuan':366, 'ga':367, 'hang':368, 'run':369, 'shua':370, 
'fo':371, 'ha':372, 'nie':373, 'pen':374, 'qia':375, 'rui':376, 'sen':377, 'chuai':378, 'nin':379, 'shei':380, 
'kei':381, 'ne':382, 'nen':383, 'pou':384, 'sou':385, 'weng':386, 'zhuai':387, 'chuo':388, 'cou':389, 'die':390, 
'jiong':391, 'keng':392, 'nang':393, 'seng':394, 'zei':395}

note_in_vocab_dic={'<pad>':0, '<bos>':1, 'wo':2, 'ni':3, 'de':4, 'yi':5, 'shi':6, 'bu':7, 'you':8, 'zai':9, 'xiang':10, 
'ai':11, 'xin':12, 'wei':13, 'zhi':14, 'ji':15, 'qing':16, 'hui':17, 'mei':18, 'dao':19, 'ren':20, 
'li':21, 'le':22, 'ye':23, 'qi':24, 'tian':25, 'guo':26, 'yao':27, 'yu':28, 'zhong':29, 'yuan':30, 
'ke':31, 'zhao':32, 'zhe':33, 'du':34, 'jiu':35, 'wu':36, 'xi':37, 'shen':38, 'sheng':39, 'shui':40, 
'he':41, 'ge':42, 'jian':43, 'jin':44, 'yan':45, 'duo':46, 'huan':47, 'zi':48, 'qu':49, 'shang':50, 
'shou':51, 'wang':52, 'ru':53, 'jing':54, 'lai':55, 'me':56, 'chu':57, 'hao':58, 'di':59, 'na':60, 
'neng':61, 'yong':62, 'bian':63, 'gan':64, 'xing':65, 'jie':66, 'hua':67, 'ming':68, 'dan':69, 'ta':70, 
'shuo':71, 'wen':72, 'hou':73, 'zhen':74, 'fen':75, 'xu':76, 'qian':77, 'rang':78, 'zui':79, 'xiao':80, 
'bi':81, 'chang':82, 'fu':83, 'xian':84, 'lian':85, 'liu':86, 'zuo':87, 'wan':88, 'kan':89, 'dui':90, 
'bei':91, 'kai':92, 'xia':93, 'fang':94, 'feng':95, 'tong':96, 'si':97, 'mo':98, 'tai':99, 'nan':100, 
'fa':101, 'ran':102, 'ci':103, 'dai':104, 'jue':105, 'bie':106, 'huo':107, 'yang':108, 'dang':109, 'cheng':110, 
'dong':111, 'nian':112, 'yin':113, 'tou':114, 'jiao':115, 'ku':116, 'dian':117, 'ban':118, 'mian':119, 'zou':120, 
'man':121, 'cong':122, 'bao':123, 'xie':124, 'lei':125, 'er':126, 'da':127, 'yue':128, 'cai':129, 'hen':130, 
'ba':131, 'liang':132, 'zen':133, 'meng':134, 'zhu':135, 'jia':136, 'jiang':137, 'deng':138, 'ting':139, 'ju':140, 
'sui':141, 'gu':142, 'que':143, 'hu':144, 'ying':145, 'ri':146, 'kong':147, 'kuai':148, 'mi':149, 'fei':150, 
'suan':151, 'quan':152, 'bai':153, 'ling':154, 'gei':155, 'pa':156, 'gai':157, 'ruo':158, 'hai':159, 'men':160, 
'shu':161, 'lu':162, 'zheng':163, 'zong':164, 'shao':165, 'ma':166, 'guan':167, 'an':168, 'ceng':169, 'fan':170, 
'gao':171, 'qiu':172, 'chen':173, 'guang':174, 'la':175, 'geng':176, 'chi':177, 'gong':178, 'pian':179, 'reng':180, 
'zao':181, 'ding':182, 'suo':183, 'huai':184, 'gou':185, 'duan':186, 'luo':187, 'ti':188, 'cuo':189, 're':190, 
'gen':191, 'yun':192, 'rong':193, 'shan':194, 'shuang':195, 'tu':196, 'xue':197, 'mu':198, 'qin':199, 'leng':200, 
'han':201, 'sha':202, 'huang':203, 'zhuan':204, 'kou':205, 'pei':206, 'diao':207, 'qie':208, 'lv':209, 'chun':210, 
'fou':211, 'san':212, 'zhan':213, 'hong':214, 'su':215, 'piao':216, 'tan':217, 'ping':218, 'bing':219, 'xun':220, 
'qiang':221, 'chuan':222, 'pang':223, 'ya':224, 'lang':225, 'se':226, 'bo':227, 'tao':228, 'she':229, 'zhang':230, 
'gui':231, 'lan':232, 'tui':233, 'zhui':234, 'chuang':235, 'chui':236, 'ou':237, 'tiao':238, 'can':239, 'po':240, 
'peng':241, 'dou':242, 'ning':243, 'zhuang':244, 'nuan':245, 'chou':246, 'gua':247, 'nei':248, 'hei':249, 'rou':250, 
'xuan':251, 'cha':252, 'lun':253, 'mang':254, 'wai':255, 'pan':256, 'tang':257, 'song':258, 'chao':259, 'chong':260, 
'miao':261, 'mou':262, 'pai':263, 'tuo':264, 'kao':265, 'lao':266, 'cang':267, 'mai':268, 'zu':269, 'ben':270, 
'hun':271, 'kuang':272, 'luan':273, 'nv':274, 'pao':275, 'a':276, 'cun':277, 'nai':278, 'chan':279, 'zhou':280, 
'lie':281, 'liao':282, 'o':283, 'ze':284, 'guai':285, 'qiao':286, 'juan':287, 'biao':288, 'nong':289, 'nao':290, 
'pu':291, 'tie':292, 'che':293, 'zan':294, 'pi':295, 'mao':296, 'e':297, 'lin':298, 'xiu':299, 'gang':300, 
'nuo':301, 'bang':302, 'ken':303, 'cui':304, 'heng':305, 'cao':306, 'nu':307, 'sa':308, 'xiong':309, 'kun':310, 
'zhuo':311, 'teng':312, 'zhun':313, 'ao':314, 'long':315, 'zeng':316, 'ruan':317, 'shun':318, 'ce':319, 'te':320, 
'zha':321, 'ca':322, 'pin':323, 'kuo':324, 'rao':325, 'shuai':326, 'min':327, 'mie':328, 'niu':329, 'bin':330, 
'en':331, 'jun':332, 'zun':333, 'lia':334, 'qiong':335, 'niang':336, 'kui':337, 'sang':338, 'lve':339, 'sao':340, 
'tun':341, 'kang':342, 'nin':343, 'chai':344, 'dun':345, 'ka':346, 'sai':347, 'wa':348, 'cu':349, 'kuan':350, 
'lou':351, 'niao':352, 'pie':353, 'zhua':354, 'kua':355, 'qun':356, 'diu':357, 'sun':358, 'tuan':359, 'zhai':360, 
'ang':361, 'gun':362, 'shai':363, 'za':364, 'ga':365, 'zang':366, 'beng':367, 'nie':368, 'shua':369, 'zuan':370, 
'hang':371, 'pen':372, 'run':373, 'ha':374, 'chuai':375, 'ne':376, 'nen':377, 'rui':378, 'sen':379, 'kei':380, 
'pou':381, 'qia':382, 'shei':383, 'sou':384, 'weng':385, 'chuo':386, 'die':387, 'fo':388, 'jiong':389, 'keng':390, 
'nang':391, 'seng':392, 'zei':393}

note_out_vocab_dic={'<pad>':0, '<bos>':1, '69':2, '74':3, '72':4, '67':5, '76':6, '71':7, '64':8, '62':9, '65':10, 
'79':11, '70':12, '60':13, '77':14, '66':15, '78':16, '73':17, '68':18, '81':19, '75':20, 
'59':21, '57':22, '63':23, '61':24, '55':25, '83':26, '80':27, '58':28, '82':29, '84':30, 
'52':31, '54':32, '56':33, '86':34, '53':35, '50':36, '85':37, '48':38, '88':39, '51':40, 
'47':41, '49':42, '45':43, '69,67':44, '76,74':45, '74,72':46, '87':47, '71,69':48, '64,62':49, '67,65':50, 
'91':51, '89':52, '78,76':53, '62,60':54, '93':55, '43':56, '66,64':57, '81,79':58, '72,74':59, '72,70':60, 
'62,64':61, '90':62, '69,71':63, '74,76':64, '46':65, '67,69':66, '69,72':67, '79,77':68, '64,66':69, '73,71':70, 
'83,81':71, '72,71':72, '74,71':73, '80,78':74, '64,67':75, '57,55':76, '65,64':77, '68,66':78, '65,67':79, '96':80, 
'67,64':81, '60,58':82, '70,72':83, '86,74':84, '44':85, '59,57':86, '76,78':87, '89,77':88, '67,70':89, '70,74':90, 
'95':91, '61,59':92, '65,63':93, '77,75':94, '62,74':95, '67,66':96, '75,73':97, '60,62':98, '55,57':99, '72,69':100, 
'63,65':101, '64,63':102, '65,77':103, '74,86':104, '77,74':105, '77,76':106, '79,67':107, '79,76':108, '62,65':109, '63,61':110, 
'69,64':111, '77,77':112, '77,89':113, '79,81':114, '81,69':115, '84,72':116, '88,76':117, '60,57':118, '64,65':119, '69,68':120, 
'70,69':121, '74,74':122, '81,83':123, '57,59':124, '60,64':125, '66,67':126, '76,79':127, '38':128, '40':129, '59,61':130, 
'60,72':131, '62,59':132, '64,61':133, '65,69':134, '73,76':135, '76,88':136, '64,69':137, '65,62':138, '76,77':139, '77,79':140, 
'79,78':141, '94':142, '65,60':143, '69,81':144, '71,71':145, '82,81':146, '84,82':147, '84,84':148, '57,60':149, '58,62':150, 
'59,62':151, '60,59':152, '62,70':153, '67,62':154, '69,69':155, '70,68':156, '71,73':157, '72,76':158, '72,84':159, '74,69':160, 
'76,71':161, '78,80':162, '80,83':163, '84,81':164, '41':165, '42':166, '50,48':167, '58,56':168, '58,60':169, '61,63':170, 
'62,61':171, '65,74':172, '67,67':173, '69,65':174, '69,70':175, '74,65':176, '74,77':177, '77,65':178, '82,70':179, '55,52':180, 
'55,60':181, '59,60':182, '60,63':183, '63,64':184, '64,76':185, '65,70':186, '66,61':187, '66,68':188, '72,72':189, '73,68':190, 
'74,73':191, '75,74':192, '78,74':193, '79,91':194, '80,81':195, '81,80':196, '86,84':197, '98':198, '39':199, '52,50':200, 
'56,54':201, '60,55':202, '62,58':203, '62,63':204, '64,64':205, '67,71':206, '71,72':207, '73,74':208, '74,62':209, '74,70':210, 
'76,75':211, '76,76':212, '78,79':213, '81,78':214, '83,80':215, '100':216, '48,47':217, '56,53':218, '58,57':219, '60,67':220, 
'61,64':221, '62,62':222, '63,60':223, '66,62':224, '70,65':225, '70,67':226, '71,67':227, '71,76':228, '71,78':229, '74,79':230, 
'75,77':231, '76,80':232, '77,69':233, '77,70':234, '79,72':235, '79,74':236, '79,82':237, '80,82':238, '82,80':239, '82,82':240, 
'83,79':241, '84,93':242, '88,86':243, '97':244, '47,50':245, '50,52':246, '51,49':247, '53,56':248, '54,52':249, '55,58':250, 
'57,58':251, '58,70':252, '59,64':253, '61,60':254, '62,57':255, '62,67':256, '63,66':257, '65,65':258, '66,65':259, '68,67':260, 
'68,69':261, '68,70':262, '69,66':263, '69,74':264, '71,66':265, '71,68':266, '71,70':267, '71,74':268, '72,75':269, '73,75':270, 
'74,78':271, '75,76':272, '76,72':273, '78,78':274, '79,79':275, '81,81':276, '82,84':277, '83,86':278, '84,76':279, '85,83':280, 
'85,85':281, '93,91':282, '99':283, '36':284, '37':285, '51,65':286, '52,55':287, '52,57':288, '55,55':289, '55,56':290, 
'56,58':291, '58,55':292, '58,64':293, '60,52':294, '60,69':295, '61,62':296, '62,55':297, '62,66':298, '63,62':299, '64,59':300, 
'64,60':301, '64,72':302, '65,58':303, '65,66':304, '65,68':305, '66,66':306, '66,69':307, '66,71':308, '67,79':309, '68,64':310, 
'68,71':311, '69,77':312, '70,70':313, '70,73':314, '72,67':315, '73,69':316, '73,70':317, '73,77':318, '74,66':319, '76,81':320, 
'77,72':321, '78,75':322, '79,65':323, '79,70':324, '81,76':325, '81,90':326, '82,83':327, '84,83':328, '84,86':329, '86,85':330, 
'89,86':331, '90,88':332, '45,47':333, '46,45':334, '47,45':335, '49,51':336, '50,47':337, '50,49':338, '50,53':339, '51,50':340, 
'52,54':341, '53,51':342, '54,63':343, '55,53':344, '57,62':345, '58,58':346, '58,61':347, '58,65':348, '59,58':349, '59,67':350, 
'60,60':351, '60,68':352, '61,56':353, '61,70':354, '62,71':355, '63,58':356, '67,72':357, '68,65':358, '69,61':359, '69,78':360, 
'69,83':361, '70,62':362, '70,71':363, '70,77':364, '70,79':365, '71,75':366, '72,60':367, '72,64':368, '72,77':369, '72,81':370, 
'73,64':371, '73,73':372, '74,60':373, '74,82':374, '75,72':375, '75,75':376, '75,85':377, '76,67':378, '76,69':379, '76,73':380, 
'76,85':381, '77,67':382, '78,69':383, '78,77':384, '79,75':385, '80,76':386, '80,77':387, '81,77':388, '81,84':389, '82,79':390, 
'83,82':391, '83,84':392, '83,85':393, '84,69':394, '85,84':395, '86,83':396, '88,85':397, '88,91':398, '91,79':399, '91,93':400, 
'99,87':401, '107':402, '38,40':403, '47,49':404, '48,50':405, '49,50':406, '50,64':407, '51,53':408, '52,45':409, '52,51':410, 
'52,53':411, '53,52':412, '53,55':413, '53,57':414, '53,58':415, '53,60':416, '54,53':417, '54,56':418, '54,57':419, '55,48':420, 
'55,59':421, '55,64':422, '55,65':423, '57,48':424, '57,56':425, '57,61':426, '58,63':427, '58,67':428, '59,52':429, '59,55':430, 
'59,68':431, '60,61':432, '60,65':433, '60,74':434, '61,61':435, '61,65':436, '62,48':437, '62,69':438, '63,59':439, '63,70':440, 
'63,85':441, '64,70':442, '64,87':443, '64,88':444, '65,61':445, '65,72':446, '66,52':447, '66,75':448, '66,76':449, '67,68':450, 
'67,74':451, '67,76':452, '68,73':453, '69,56':454, '69,59':455, '69,62':456, '69,84':457, '70,58':458, '70,60':459, '70,66':460, 
'70,76':461, '71,59':462, '71,62':463, '71,63':464, '71,80':465, '71,82':466, '72,63':467, '72,65':468, '72,66':469, '72,73':470, 
'72,80':471, '73,72':472, '73,78':473, '74,64':474, '74,67':475, '74,75':476, '75,78':477, '75,84':478, '76,64':479, '76,84':480, 
'76,87':481, '77,78':482, '77,81':483, '77,82':484, '78,71':485, '78,81':486, '78,83':487, '78,86':488, '79,52':489, '79,69':490, 
'79,71':491, '79,80':492, '80,79':493, '81,73':494, '81,74':495, '81,82':496, '81,86':497, '82,72':498, '82,74':499, '82,77':500, 
'83,49':501, '83,83':502, '83,87':503, '83,91':504, '84,77':505, '84,88':506, '85,64':507, '85,86':508, '85,87':509, '85,88':510, 
'85,93':511, '86,77':512, '86,82':513, '86,90':514, '87,63':515, '87,64':516, '87,87':517, '87,90':518, '87,95':519, '88,79':520, 
'90,87':521, '90,93':522, '91,88':523, '91,95':524, '92':525, '92,90':526, '93,88':527, '95,83':528, '95,91':529}


def translate_sentence(sentence, note_in_vocab, note_out_vocab, model, device, max_len):
    PAD, BOS = '<pad>', '<bos>'
    model.eval()
    # in_tokens = sentence.split(' ')

    p = Pinyin()
    biaodian = '，。？！：；、“”‘’《》,.?!:; '
    in_tokens = []
    for k in sentence:
        if k in biaodian:
            continue
        in_tokens.append(p.get_pinyin(k, ''))

    real_len = len(in_tokens)
    if real_len > max_len:
        print("seq is longer than max_len!")
        return

    in_tokens += [PAD] * (max_len - len(in_tokens))
    enc_input = torch.tensor([note_in_vocab[tk] for tk in in_tokens])
    src_tensor = torch.LongTensor(enc_input).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [note_out_vocab[BOS]]
    for i in range(real_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output[:, :, 2:-1].argmax(2)[:, -1].item()
        pred_token += 2

        trg_indexes.append(pred_token)

    out_vocab_reverse = {v: k for k, v in note_out_vocab.items()}
    trg_tokens = [out_vocab_reverse[int(i)] for i in trg_indexes]
    return trg_tokens[1:]


def generate_note_dura(input_seq):
    # input_seq为输入的宋词
    # note_translation, dura_translation分别为生成的note序列和duration序列，以列表形式返回
    # 需要调用translate_sentence函数
    max_seq_len = 128
    device = 'cpu'
    note_enc = Encoder(394, 270, 3, 6, 300, 0.15, device, max_seq_len)
    note_dec = Decoder(530, 270, 3, 6, 320, 0.16, device, max_seq_len)
    dura_enc = Encoder(396, 270, 3, 6, 300, 0.15, device, max_seq_len)
    dura_dec = Decoder(45, 270, 3, 6, 288, 0.16, device, max_seq_len)
    SRC_PAD_IDX = 0
    TRG_PAD_IDX = 0
    note_model = Seq2Seq(note_enc, note_dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    dura_model = Seq2Seq(dura_enc, dura_dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    dura_model = torch.load('./final5_dura_model_0.pt', map_location='cpu')  #test:0.00011214
    note_model = torch.load('./final5_note_model_0.pt', map_location='cpu')  #test:0.0017557
    note_model.encoder.device = 'cpu'
    note_model.decoder.device = 'cpu'
    dura_model.encoder.device = 'cpu'
    dura_model.decoder.device = 'cpu'
    note_model.device = 'cpu'
    dura_model.device = 'cpu'

    note_translation = translate_sentence(input_seq, note_in_vocab_dic, note_out_vocab_dic, note_model, device,
                                          max_seq_len)
    dura_translation = translate_sentence(input_seq, dura_in_vocab_dic, dura_out_vocab_dic, dura_model, device,
                                          max_seq_len)
    result = []
    tokens_length = len(input_seq)
    ly = ''
    for i in range(tokens_length):
        if input_seq[i] in '，。！？；、':
            ly = ly[0:-1]
            if i != tokens_length - 1:
                ly += '&#&'
        elif input_seq[i] in '，。？！：；、“”‘’《》,.?!:; ':
            continue
        else:
            ly += input_seq[i]
            ly += ' '
    lyrics = ly.split('&#&')
    result = []
    x = 0
    for seg in lyrics:
        temp = (len(seg) + 1) // 2
        dic = {}
        dic['lyrics'] = seg
        dic['key'] = note_translation[x:x + temp]
        dic['duration'] = dura_translation[x:x + temp]
        result.append(dic)
        x = x + temp
    return result


# 使用方法范例（可能会出现警告SourceChangeWarning，不影响使用）
input_seq = '碧云天，黄叶地，秋色连波，波上寒烟翠。山映斜阳天接水，芳草无情，更在斜阳外。黯乡魂，追旅思，夜夜除非，好梦留人睡。明月楼高休独倚，酒入愁肠，化作相思泪。'
n_d=generate_note_dura(input_seq)
print(n_d)
# 输出
# [{'lyrics': '碧 云 天', 'key': ['71', '69', '71'], 'duration': ['0.3', '0.5', '0.5']}, {'lyrics': '黄 叶 地', 'key': ['69', '67', '66'], 'duration': ['0.5', '0.3', '0.3']}, {'lyrics': '秋 色 连 波', 'key': ['64', '67', '66', '67'], 'duration': ['0.5', '0.3', '1.3', '0.5']}, {'lyrics': '波 上 寒 烟 翠', 'key': ['69', '71', '69', '67', '66'], 'duration': ['0.5', '0.3', '0.3', '1.3', '0.3']}, {'lyrics': '山 映 斜 阳 天 接 水', 'key': ['64', '67', '69', '71', '69', '71', '62'], 'duration': ['0.5', '0.3', '0.8', '0.3', '0.3', '0.5', '0.3']}, {'lyrics': '芳 草 无 情', 'key': ['64', '64', '67', '66'], 'duration': ['0.3', '0.5', '0.5', '0.3']}, {'lyrics': '更 在 斜 阳 外', 'key': ['62', '71', '74', '74', '71'], 'duration': ['0.3', '1.3', '0.5', '0.5', '0.3']}, {'lyrics': '黯 乡 魂', 'key': ['69', '71', '71'], 'duration': ['0.3', '0.3', '0.5']}, {'lyrics': '追 旅 思', 'key': ['71', '71', '76'], 'duration': ['0.3', '0.8', '0.3']}, {'lyrics': '夜 夜 除 非', 'key': ['74', '71', '74', '71'], 'duration': ['0.3', '0.5', '0.3', '0.8']}, {'lyrics': '好 梦 留 人 睡', 'key': ['74', '76', '78', '76', '74'], 'duration': ['0.3', '0.5', '0.3', '0.3', '1.3']}, {'lyrics': '明 月 楼 高 休 独 倚', 'key': ['71', '69', '67', '67', '69', '69', '67'], 'duration': ['0.5', '0.3', '0.3', '0.3', '0.5', '0.5', '0.3']}, {'lyrics': '酒 入 愁 肠', 'key': ['71', '72', '72', '71'], 'duration': ['0.3', '0.3', '0.5', '0.3']}, {'lyrics': '化 作 相 思 泪', 'key': ['69', '67', '71', '69', '67'], 'duration': ['0.3', '0.5', '0.3', '0.3', '0.3']}]
