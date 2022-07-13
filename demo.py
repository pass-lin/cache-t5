# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:54:14 2022

@author: admin
"""
#基于bert4keras的t5 cache
#演示使用的是t5 PEGASUS
base_path='chinese_t5_pegasus_base/'
config_path = base_path+'config.json'
checkpoint_path = base_path+'model.ckpt'
dict_path = base_path+'vocab.txt'
import os
#因为是tf2.x，所以使用的是tf.keras
os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cahceT5 import T5_Decoder_cache
import jieba
from bert4keras.models import *
from bert4keras.tokenizers import Tokenizer
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)
class my_decoder(T5_Decoder_cache):
    def cache_compute(self, c,start_token, end_token,maxlen,length=1100):
        #length就是如果你没有初始化位置编码cache，初始化长度为length的位置编码cache
        #提供了一个单条数据预测的greedy演示
        #在你使用该cache时可以根据需要重写该方法
        x=[start_token]*c.shape[0]#获取开始token
        x=np.reshape(x,[-1,1])
        outputs=[]    
        cache_dict={}#cache字典
        c_len=K.shape(c)[1]
        #x[0][0]!=end_token and
        while  len(outputs)<maxlen and x[0][0]!=end_token:
            c,x=self.apply_embeddings([c,x])#计算embeding
            if len(cache_dict)>0:
                #第一次预测之后的调用位置编码
                self.have_cache_input=True
                a=cache_dict['Decoder-Transformer-0-MultiHeadSelfAttentionv']
                self.position_bias = self.get_position_bias_cache([a, c],length)
            else:
                #第一次预测调用位置编码
                self.have_cache_input=False
                self.position_bias = self.get_position_bias_cache([x, c],length)
            #transformer
            for index in range(self.num_hidden_layers):
                c,x=self.apply_main_layers([c,x],index,cache_dict)
            #输出
            x=self.apply_final_layers([c,x])
            
            #贪心搜索，可以根据需要自己修改
            x=np.argmax(x,-1)
            #打印一下输出
            print(tokenizer.decode(x[0]),end='')
            outputs.append(x[0][0])
        return outputs
    
encoder = build_transformer_model(
    config_path=config_path,
    #checkpoint_path=checkpoint_path,不读ckpt是因为我用的是自己训练好的summary.h5
    model='mt5.1.1_encoder',
    return_keras_model=True,
)
encoder.load_weights('summary.h5',by_name=True)
decoder=build_transformer_model(
    config_path=config_path,
    #checkpoint_path=checkpoint_path,
    model=my_decoder,
    return_keras_model=False,
    version='mt5.1.1',
    with_lm='linear',
    
)
decoder.model.load_weights('summary.h5',by_name=True)
#使用之前要初始化位置编码的cache，取encoder和decoder最大长度的较大值
decoder.initial_position_bias_cache(1024)
#一个简单的使用demo
word='''正所谓千军易得一将难求，自古以来，带兵打仗的将军在一定程度上能够改变一场战争的胜负。

朝鲜战争爆发后，毛主席等中央大佬未雨绸缪，提前做好了最坏的打算。先是由粟裕挂帅，成立了东北边防军，以应对当前的局势。
随着朝鲜战争局势的进一步恶化，战火已经烧到了我国边境，应斯大林和金日成的请求，我国决定派遣志愿军入朝，保家卫国，​抗美援朝！

​那么，究竟让谁来担任支援的统帅合适呢？关于人选方面，毛主席最初的选择并非​彭德怀，而是粟裕与林彪。

众所周知，粟裕和林彪是解放战争时期四野和三野的军事主官（三野粟裕负责军事指挥），​两大野战军战绩出色，歼敌人数多，为解放战争的胜利​立下大功。

既然​毛主席最初让粟裕担任东北边防军的司令员兼政委，他自然成了入朝指挥作战的第一人。
可粟裕自解放战争后，身体情况一直不太理想，身上有弹片，患有美尼尔氏综合征，长期处在养病状态，无法主持工作。

毛主席给了粟裕几个月的休养时间，在依旧没有好转的情况下，只能另外选人。

粟裕因病无法带兵入朝作战，林彪成了最合适的人选，可当时他身体也不好，处于养病状态。

再加上林彪认为中美实力差距较大，发生战争对我国经济建设极为不利，​他是反对出兵的。

​实事求是地说，林彪这种观点也没有错，只是看法不同，都是为了新中国。当时与​林彪看法一致的人大有人在，因此没有必要为了这件事对他有不好的印象。

尽管抗美援朝时期林彪并没有直接带兵作战，但他与周总理一起前往苏联与斯大林谈军事援助的问题，​作出了应有的贡献。
既然粟裕和林彪都因病无缘挂帅，那么毛主席必须再考虑其他人选，就在他犯难的时候，陈毅主动请缨，​为主席分忧。

陈毅是十大元帅之一，综合能力突出，军事能力虽然不及粟裕与林彪，但非常擅长搞统筹协调工作，能够很好地团结内部关系，是一位善于人际交往的元帅。

毛主席非常欣赏陈毅这种在关键时刻，主动请缨挑重担的行为，对他进行了表扬。

可当时的陈毅是上海市市长，上海是国际大都市，地位突出且形势复杂，一般人无法掌控，陈毅在上海工作多年，工作成绩很好，熟悉那里的情况，​一时间无法离开。
鉴于这方面的考虑，毛主席最终婉拒了陈毅的请求，没有让他​入朝指挥作战。

早在粟裕和林彪无法领兵指挥抗美援朝的时候，毛主席心中就有一个合适的人选，他就是彭德怀。

彭德怀是一位作战经丰富的统帅，新中国十大元帅排名第二，​在军中的地位仅次于朱德。

从红军时期开始，彭德怀与毛主席就有着非常深厚的情谊。一次，彭德怀打了胜仗，毛主席写了“谁敢横刀立马，唯我彭大将军”​夸赞彭德怀。

当时的彭德怀在西北，主持西北的建设开发工作，得知中央让他马上进京时，彭德怀还以为毛主席等人要找他谈西北的建设问题，把规划图都带上了飞机，​正准备汇报工作。

彭德怀的飞机一到北京，毛主席正在与中央委员们开会，讨论​是否出兵抗美援朝的事情。
由于彭德怀刚到不久，对局势不是非常了解，在大会上​并没有发言。会议结束后，彭德怀觉得中国必须出兵，不能等战火烧到中国再打，这样损失更大。

彭德怀支持出兵，与毛主席的意见一致，二人不谋而合。

得知彭德怀的态度后，毛主席在大会上提议彭德怀​担任志愿军司令员兼政治委员，得到了绝大多数人，包括朱德、周恩来等人的支持。

既然是中央的决定，彭德怀义无反顾，为了国家和民族，他​勇担重任，踏上了前往朝鲜的征途。

在抗美援朝战争中，彭德怀指挥志愿军打了五次大的战役，将战线从鸭绿江边推到了三八线附近，给了以美国为首的“联军”重创，​取得了抗美援朝战争的伟大胜利。
彭德怀在抗美援朝中的贡献自然不用多说，他也一跃成为世界顶级名将，击败了二战名将麦克阿瑟等美军高级将领。

从此享誉世界，让世人记住了中国有一位名叫彭德怀的人，农民出身，没有受过正规的军事教育，却能打败强大的美军​。​

正如彭德怀所说：

抗美援朝雄辩证明，西方侵略者几百年来只要在东方的一个海岸上架起几尊大炮就可以霸占一个国家的时代是一去不复返了。
'''
#编码
token=tokenizer.encode(word,maxlen=1024)[0]
token=np.reshape(token,[1,-1])
c=encoder(token)
#解码
y=decoder.cache_compute(c,tokenizer._token_start_id,tokenizer._token_end_id,maxlen=30)
