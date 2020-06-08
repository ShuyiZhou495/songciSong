from generate_note_dura import generate_note_dura
from generate_note_dura import Seq2Seq
from generate_note_dura import Encoder
from generate_note_dura import EncoderLayer
from generate_note_dura import MultiHeadAttentionLayer
from generate_note_dura import PositionwiseFeedforwardLayer
from generate_note_dura import Decoder
from generate_note_dura import DecoderLayer

if __name__ == '__main__':
    input_seq = '碧云天，黄叶地，秋色连波，波上寒烟翠。山映斜阳天接水，芳草无情，更在斜阳外。黯乡魂，追旅思，夜夜除非，好梦留人睡。明月楼高休独倚，酒入愁肠，化作相思泪。'
    print(generate_note_dura(input_seq))