# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:32:37 2022

@author: Administrator
"""
from tqdm import tqdm
import numpy as np
from bert4keras.models import *
import os
class FeedForward_simple(FeedForward):
    #因为只做mt5，所以把if 省略了
    def call(self, inputs):
        x = self.i0_dense(inputs)*self.i1_dense(inputs)
        x = self.o_dense(x)
        return x
class RelativePositionEmbeddingT5_cache(RelativePositionEmbeddingT5):
    #对苏神的T5做一下cache的简化
    def __init__(self,gather=True,**kwargs):
        super(RelativePositionEmbeddingT5_cache, self).__init__(**kwargs)
        self.gather=gather
    def build(self, input_shape):
        super(RelativePositionEmbeddingT5_cache, self).build(input_shape)
        self.indexs=self.compute_position_ids(3000,3000)#预存储索引提升速度
    def call(self,inputs):
        #大于预存储的索引时就算
        q_len,v_len=inputs[:]
        pos_ids = self.compute_position_ids(q_len,v_len)
        if q_len<3000 and v_len<3000:
            pos_ids =self.indexs[:q_len,:v_len]
        else:
            pos_ids=K.gather(pos_ids,[K.shape(pos_ids)[0]-1])
        if self.gather:
            pos_ids=K.gather(pos_ids, [K.shape(pos_ids)[0]-1])
        return K.gather(self.embeddings, pos_ids)
    def compute_position_ids(self, q_len,v_len):
        """T5的相对位置分桶（直接翻译自官方T5源码）
        但略作修改，输入是q和v的长度
        """
        # 计算位置差
        q_idxs = K.arange(0, q_len, dtype='int32')
        q_idxs = K.expand_dims(q_idxs, 1)
        v_idxs = K.arange(0, v_len, dtype='int32')
        
        v_idxs = K.expand_dims(v_idxs, 0)
        pos_ids = v_idxs - q_idxs
        # 后处理操作
        num_buckets, max_distance = self.input_dim, self.max_distance
        ret = 0
        n = -pos_ids
        if self.bidirectional:
            num_buckets //= 2
            ret += K.cast(K.less(n, 0), 'int32') * num_buckets
            n = K.abs(n)
        else:
            n = K.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = K.less(n, max_exact)
        val_if_large = max_exact + K.cast(
            K.log(K.cast(n, K.floatx()) / max_exact) /
            np.log(max_distance / max_exact) * (num_buckets - max_exact),
            'int32',
        )
        val_if_large = K.minimum(val_if_large, num_buckets - 1)
        ret += K.switch(is_small, n, val_if_large)
        
        return ret
    def compute_mask(self, inputs, mask):
        return mask

class MultiHeadAttention_cache(MultiHeadAttention):
    def __init__(self,cross_flag=False,**kwargs):
        super(MultiHeadAttention_cache, self).__init__(**kwargs)
        self.cross_flag=cross_flag#判断是不是cross attention
    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力的cache版本
        """
        q, k, v = inputs[:3]
        k_cache=kwargs.get('k_cache')#没有cache就算None
        v_cache=kwargs.get('v_cache')
        # 线性变换
        qw = self.q_dense(q)

        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        qw=tf.transpose(qw,[0,2,1,3])
        #cross attention只在第一次做计算
        if k_cache==None or self.cross_flag==False:
            kw = self.k_dense(k)
            vw = self.v_dense(v)
            
        if k_cache==None:
            kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
            vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
            kw=tf.transpose(kw,[0,2,3,1])
            vw=tf.transpose(vw,[0,2,1,3])
        # Attention
        elif self.cross_flag:
            
            kw=k_cache
            vw=v_cache
        else:
            kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
            vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
            
            kw=tf.transpose(kw,[0,2,3,1])
            vw=tf.transpose(vw,[0,2,1,3])
            
            kw=tf.concat([k_cache,kw],axis=-1)
            vw=tf.concat([v_cache,vw],axis=-2)
        inputs = [qw, kw, vw] + inputs[3:]
        #pay attention
        (qw, kw, vw), n = inputs[:3], 3
        p_bias =  kwargs.get('p_bias')
        # Attention
        a=qw@kw
        # 处理位置编码
        position_bias = K.permute_dimensions(inputs[n], (2, 0, 1))
        a = a + K.expand_dims(position_bias, 0)
        
        

        A = attention_normalize(a, -1, self.normalization)
        # 完成输出
        o = A@vw
        o=tf.transpose(o,[0,2,1,3])
        # 完成输出
        o = K.reshape(o, (-1, K.shape(o)[1], self.head_size * self.heads))
        #end
        o = self.o_dense(o)
        # 返回结果
        
        return [o,kw,vw]
class T5_Decoder_cache(T5_Decoder):
    """Google的T5模型（Decoder）
    """
    def __init__(self, have_cache_input=True, **kwargs):
        super(T5_Decoder_cache, self).__init__(**kwargs)
        self.have_cache_input = False#现在有没有cache输入的标志
        self.postion_cache=None#预存储的位置编码
    
    def initial_position_bias_cache(self,length1=50,length2=None):
        #先把位置编码存下来，避免重复索引浪费时间
        if length2==None:
            length2=length1
        self.postion_cache=[]
        print('initial position cache')
        for i in tqdm(range(1,length1+1)):
            position=self.apply(
                inputs=[i, length2],
                layer= RelativePositionEmbeddingT5_cache,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                gater=False,
                name='Decoder-Embedding-Relative-Position'
            )
            self.postion_cache.append(position.numpy())
        self.postion_cache=np.squeeze(self.postion_cache,1)
        self.postion_cache=tf.constant(self.postion_cache)
        self.position_bias=None
    def get_position_bias_cache(self,inputs,length=1100):
        #获取存储好的位置编码
        x, c = inputs
        x_len=x.shape[-2]
        c_len=c.shape[1]
        if self.have_cache_input:
            x_len+=1
        if self.postion_cache==None:
            
            self.initial_position_bias_cache(length)
        if x_len>length or c_len>length:
            self.position_bias=None
            return self.compute_position_bias(inputs)
        
        p1=self.postion_cache[x_len-1:x_len,:x_len]
        p2=self.postion_cache[x_len-1:x_len,:c_len]
        return [p1,p2]
        
    def cache_compute(self, c,start_token, end_token,maxlen,length=1100):
        pass
    def apply_embeddings(self, inputs):
        """T5的embedding只有token embedding，
        
        """
        c, x = inputs

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=False,
            name='Embedding-Token'
        )

        return [c, x]
    def apply_main_layers(self, inputs, index,cache_dict=None,position_bias=None):
        """T5的Decoder主体是基于Self-Attention、Cross-Attention的模块
        顺序：LN --> Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add
        """
        self_attention_name = 'Decoder-Transformer-%d-MultiHeadSelfAttention' % index
        cross_attention_name = 'Decoder-Transformer-%d-MultiHeadCrossAttention' % index
        feed_forward_name = 'Decoder-Transformer-%d-FeedForward' % index
        c, x = inputs
        z = self.layer_norm_conds[0]
        #如果没有cache_dict是keras构建model的时候
        
        
        if cache_dict!=None:
            k_cache_self,v_cache_self=cache_dict.get(self_attention_name+'k'),cache_dict.get(self_attention_name+'v')
            k_cache_cross,v_cache_cross=cache_dict.get(cross_attention_name+'k'),cache_dict.get(cross_attention_name+'v')
            if k_cache_self==None:
                self.have_cache_input=False
            else:
                self.have_cache_input=True
        #获取位置编码，不能直接用get_position_bias_cache这样子keras会编译不通过
        #因为self.compute_position_bias在编译的时候返回的是对应层的计算结果，在推理的时候返回的是self.position_bias
        #所以在推理的时候通过self.position_bias=get_position_bias_cache可以曲线救国
        if self.have_cache_input:
            position_bias = self.compute_position_bias([k_cache_self, c])
        else:
            position_bias = self.compute_position_bias([x, c])
        
        # Self Attention
        
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % self_attention_name
        )
        
        #有没有cache的两种输入
        if self.have_cache_input:
            x,kw, vw = self.apply(
                inputs=[x, x, x,  position_bias[0]],
                layer=MultiHeadAttention_cache,
                arguments={
                    'a_bias': False,
                    'p_bias': 't5_relative',
                    'k_cache':k_cache_self,
                    'v_cache':v_cache_self,
                },
                heads=self.num_attention_heads,
                head_size=self.attention_head_size,
                out_dim=self.hidden_size,
                key_size=self.attention_key_size,
                use_bias=False,
                attention_scale=False,
                attention_dropout=self.attention_dropout_rate,
                kernel_initializer=self.initializer,
                name=self_attention_name
            )
        else:
            
            x,kw, vw = self.apply(
                inputs=[x, x, x,  position_bias[0]],
                layer=MultiHeadAttention_cache,
                arguments={
                    'a_bias': False,
                    'p_bias': 't5_relative'
                },
                heads=self.num_attention_heads,
                head_size=self.attention_head_size,
                out_dim=self.hidden_size,
                key_size=self.attention_key_size,
                use_bias=False,
                attention_scale=False,
                attention_dropout=self.attention_dropout_rate,
                kernel_initializer=self.initializer,
                name=self_attention_name,
                
            )
        if cache_dict!=None:
            #推理时用的加入cache字典
            cache_dict[self_attention_name+'k']=kw
            cache_dict[self_attention_name+'v']=vw
        
        x=x+xi
        
        
        
        # Cross Attention
        xi = x
        
            
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % cross_attention_name
        )
        if self.have_cache_input:
             x,kw,vw = self.apply(
                inputs=[x, c, c, position_bias[1]],
                layer=MultiHeadAttention_cache,
                arguments={
                    'a_bias': None,
                    'p_bias': 't5_relative',
                    'k_cache':k_cache_cross,
                    'v_cache':v_cache_cross,
                },
                heads=self.num_attention_heads,
                head_size=self.attention_head_size,
                out_dim=self.hidden_size,
                key_size=self.attention_key_size,
                use_bias=False,
                attention_scale=False,
                attention_dropout=self.attention_dropout_rate,
                kernel_initializer=self.initializer,
                cross_flag=True,
                name=cross_attention_name
                
            )
        else:
            x,kw,vw = self.apply(
                inputs=[x, c, c, position_bias[1]],
                layer=MultiHeadAttention_cache,
                arguments={
                    'a_bias': None,
                    'p_bias': 't5_relative'
                },
                heads=self.num_attention_heads,
                head_size=self.attention_head_size,
                out_dim=self.hidden_size,
                key_size=self.attention_key_size,
                use_bias=False,
                attention_scale=False,
                attention_dropout=self.attention_dropout_rate,
                kernel_initializer=self.initializer,
                cross_flag=True,
                name=cross_attention_name
            )
        if cache_dict!=None:
            cache_dict[cross_attention_name+'k']=kw
            cache_dict[cross_attention_name+'v']=vw

        x=x+xi

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            zero_mean=False,
            offset=False,
            epsilon=1e-6,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=FeedForward_simple,
            units=self.intermediate_size,
            activation=self.hidden_act,
            use_bias=False,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x=x+xi

        return [c, x]
    def compute_position_bias(self, inputs=None):
        """T5相对位置编码
        """
        if self.position_bias is None:

            x, c = inputs
            x_len=K.shape(x)[1]
            c_len=K.shape(c)[1]
            #如果有cache输入，那么实际位置应该+1
            if self.have_cache_input:
                x_len+=1
            p1 = self.apply(
                inputs=[x_len, x_len],
                layer= RelativePositionEmbeddingT5_cache,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            p2 = self.apply(
                inputs=[x_len, c_len],
                layer= RelativePositionEmbeddingT5_cache,
                input_dim=32,
                output_dim=self.num_attention_heads,
                bidirectional=False,
                embeddings_initializer=self.initializer,
                name='Decoder-Embedding-Relative-Position'
            )
            self.position_bias = (p1, p2)

        return self.position_bias
