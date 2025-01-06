# -*- coding: utf-8 -*-
import os
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel,BertTokenizer, BertForMaskedLM

def gelu(x):
    return x  * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class knowMP(nn.Module):
    def __init__(self, B, N, K, max_length, data_dir):
        nn.Module.__init__(self)

        self.batch = B
        self.n_way = N
        self.k_shot = K
        self.max_length = max_length
        self.data_dir = data_dir
        self.hidden_size = 768 # bert-base  

        self.cost = nn.NLLLoss() #损失函数
        self.coder = BERT(N,max_length,data_dir) #元编码器和软模板设计
        self.initializer = Initializer(N,K, data_dir) #标签词初始化
        
        self.W = [None] * self.batch # label word embedding matrix
        print()
    def loss(self,logits,label):
        return self.cost(logits.log(),label.view(-1)) 

    def accuracy(self,logits,label):
        label = label.view(-1)
        _, pred = torch.max(logits,1) #预测
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
    #标签词快速调优
    def forward(self, inputs, W): # inputs: [N*K or total_Q, hidden_size] 
                                  # W: [n_way, hidden_size]  
        # s2w
        logits_for_instances = F.linear(inputs, W, bias=None) #[N*K or total_Q or 1 ,n_way]
        # w2s
        if inputs.shape[0] > 1:
            logits_for_classes = F.linear(W, torch.mean(inputs.view(self.n_way, inputs.shape[0]//self.n_way,768),dim=1), bias=None)
            return F.softmax(logits_for_instances,dim=-1), F.softmax(logits_for_classes,dim=-1)
        else:
            return F.softmax(logits_for_instances,dim=-1), F.softmax(logits_for_instances,dim=-1)

    def get_info(self,class_names): # list of class_name
        return self.initializer.get_embedding(class_names) # [N * [candidate word embeddings]]

    def prework(self,candidate_word_embeddings): # meta-info: [N, hidden_size]
                                                # support:   [N*K, bert_size]
        return self.initializer(candidate_word_embeddings)

class Initializer(nn.Module):#标签词初始化
    def __init__(self, N, K, data_dir):
        super(Initializer,self).__init__()
        self.n_way = N
        self.k_shot = K
        self.embedding_dim = 768

        candidate_info = data_dir['candidates'] # candidate word info
        if candidate_info is None or not os.path.isfile(candidate_info):
            raise Exception("[ERROR] candidate words information file doesn't exist")

        self.cl2embed = json.load(open(candidate_info,'r')) # {class_name: candidate word embeddings}

        for key in self.cl2embed.keys():
            self.cl2embed[key] = torch.Tensor(self.cl2embed[key]).cuda()

    def get_embedding(self, class_names):
        # read candidate word embeddings from the class name 
        res = []
        for i in range(len(class_names)): 
            class_name = class_names[i]
            vec_list = self.cl2embed[class_name].float()
            res.append(vec_list) 
        return res  # [N * [candidate word embeddings]]

    def forward(self, inputs): # inputs: [N * [candidate word embeddings]]
        # average pooling
        W = torch.zeros(len(inputs), self.embedding_dim).cuda()
        for idx in range(len(inputs)):
            W[idx] = torch.mean(inputs[idx], 0).requires_grad_(True) # [hidden_size] candidates mean pooler
            # W[idx] = inputs[idx][0].requires_grad_(True) # [hidden_size] without kg
        if self.k_shot == 1:    #标准化处理
            W = F.normalize(W,dim=-1)
        elif self.k_shot == 5:
            W = 0.5 * F.normalize(W,dim=-1)

        return W
               
class BERT(nn.Module):#元编码器和模板设计
    def __init__(self, N, max_length, data_dir, blank_padding=True):
        super(BERT,self).__init__()
        self.cuda = torch.cuda.is_available()
        self.n_way = N
        self.max_length = max_length
        self.blank_padding = blank_padding
        # print(os.listdir('./bert-base-uncased/'))
        self.pretrained_path = './bert-base-uncased/'

        bert_model = BertModel.from_pretrained(self.pretrained_path)
        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
        self.bert_ebd = bert_model.embeddings
        self.bert_encoder = bert_model.encoder

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        self.dropout = nn.Dropout(data_dir['pb_dropout'])
        self.benchmark = data_dir['benchmark']
        
        mlm = BertForMaskedLM.from_pretrained(self.pretrained_path)
        D = mlm.cls.state_dict()
        (pred_bias, tf_dw, tf_db, tf_lnw, tf_lnb, dec_w, dec_b) = (D['predictions.bias'],
                                                            D['predictions.transform.dense.weight'],
                                                            D['predictions.transform.dense.bias'], 
                                                            D['predictions.transform.LayerNorm.weight'], 
                                                            D['predictions.transform.LayerNorm.bias'], 
                                                            D['predictions.decoder.weight'],
                                                            D['predictions.decoder.bias'])
        self.LayerNorm = nn.LayerNorm(768,eps = 1e-12) 
        self.LayerNorm.weight.data, self.LayerNorm.bias.data = tf_lnw,tf_lnb
        self.tf_dense = nn.Linear(768,768)
        self.tf_dense.weight.data,self.tf_dense.bias.data = tf_dw,tf_db

        # soft template params
        if self.benchmark == "FewRel":
            self.soft_prompt = nn.Parameter(torch.rand(21,768)) #创建可训练参数，在模型训练过程中自动更新
            # soft_token = ['is', '[MASK]', 'of', '.']   #离散的token-ids
            # soft_token = ["is", "a", "entity", ",", ".", "the", "relation", "between", "and","is", "[MASK]", "."]
            soft_token = ['in', 'this', 'sentence', ',', 'is', 'an', 'entity', 'of', 'type', ',', 'and', 'is', 'an', 'entity', 'of', 'type', '.', 'is', '[MASK]', 'of', '.']
            # soft_token = ['the', 'relation', 'between', 'entity', 'with', 'entity', 'type', 'and', 'entity', 'type', 'is', '[MASK]', '.']
            # soft_token = ['the', 'relation', 'between', 'entity', 'with', 'entity', 'type', 'and', 'entity', 'with', 'entity', 'type', 'is', '[MASK]', '.']

        
        soft_token_id = self.tokenizer.convert_tokens_to_ids(soft_token)   #将单词转换成词汇表中的id
        for i in range(len(soft_token)):#
            self.soft_prompt.data[i] = self.bert_ebd.word_embeddings.weight.data[soft_token_id[i]] #将离散的token-ids映射到单词嵌入中

    def forward(self,inputs):
        if self.benchmark == "FewRel":
            return self.forward_FewRel(inputs)


    def forward_FewRel(self,inputs): # [raw_tokens_dict * (N*K or total_Q)]
        input_ebds, MASK_INDs,att_masks,outputs = [],[],[],[]
        for _ in inputs:
            # indexed_token, indexed_head, indexed_tail, avai_len = self.tokenize_FewRel(_)

            indexed_token, indexed_head, indexed_tail, indexed_head_type, indexed_tail_type, avai_len = self.tokenize_FewRel(_)

            after_ebd_text = self.bert_ebd.word_embeddings(indexed_token) # [1,avai_len] ——> [1, avai_len, 768]
            after_ebd_head = self.bert_ebd.word_embeddings(indexed_head)  # [1,len_head] ——> [1, len_head, 768]
            after_ebd_tail = self.bert_ebd.word_embeddings(indexed_tail)  # [1,len_tail] ——> [1, len_tail, 768]
            after_ebd_head_type = self.bert_ebd.word_embeddings(indexed_head_type) #  [1, len_head_type] ——> [1, len_head_type, 768]
            after_ebd_tail_type = self.bert_ebd.word_embeddings(indexed_tail_type)#  [1,len_tail_type] ——> [1, len_tail_type, 768]

            # input_ebd = torch.cat((after_ebd_text, after_ebd_head, self.soft_prompt[:3].unsqueeze(0)),
            #                       1)  # text head is [mask] of (embedding)

            # MASK_INDs.append(avai_len + indexed_head.shape[-1] + 1)


            input_ebd = torch.cat((after_ebd_text,
                                   self.soft_prompt[:4].unsqueeze(0),
                                   after_ebd_head, self.soft_prompt[4:9].unsqueeze(0), after_ebd_head_type, self.soft_prompt[9:11].unsqueeze(0),
                                   after_ebd_tail, self.soft_prompt[11:16].unsqueeze(0), after_ebd_tail_type, self.soft_prompt[16].unsqueeze(0).unsqueeze(0),
                                   after_ebd_head, self.soft_prompt[17:20].unsqueeze(0),
                                   after_ebd_tail, self.soft_prompt[20].unsqueeze(0).unsqueeze(0),
                                   self.bert_ebd.word_embeddings(torch.tensor(102).cuda()).unsqueeze(0).unsqueeze(0)
                                   ),1)
            MASK_INDs.append(avai_len + 2*indexed_head.shape[-1] + indexed_tail.shape[-1]+
                             indexed_head_type.shape[-1]+indexed_tail_type.shape[-1]+18)
            
            

            # mask calculation
            att_mask = torch.zeros(1,self.max_length)
            if self.cuda: att_mask = att_mask.cuda()
            att_mask[0][:input_ebd.shape[1]]=1 # [1, max_length]

            # padding tensor
            while input_ebd.shape[1] < self. max_length:
                input_ebd = torch.cat((input_ebd, self.bert_ebd.word_embeddings(torch.tensor(0).cuda()).unsqueeze(0).unsqueeze(0)), 1)

            input_ebd = input_ebd[:,:self.max_length,:]
            input_ebds.append(input_ebd)

            input_shape = att_mask.size()
            device = indexed_token.device
            
            extented_att_mask = self.get_extended_attention_mask(att_mask, input_shape,device) #扩展注意力掩码，用于指示哪部分应该被关注，哪部分应该被忽略
            att_masks.append(extented_att_mask)

        input_ebds = torch.cat(input_ebds,0) # [N*K, max_length，768]
        tensor_masks = torch.cat(att_masks,0) # [N*K, max_length] then extend
        sequence_output= self.bert_encoder(self.bert_ebd(inputs_embeds = input_ebds) , attention_mask = tensor_masks).last_hidden_state # 编码[N*K, max_length, bert_size]

        
        for i in range(input_ebds.size(0)): 
            outputs.append(self.entity_start_state(MASK_INDs[i],sequence_output[i]))
            # [[1,bert_size*2] * (N*K)]
        tensor_outputs = torch.cat(outputs,0)  # [N*K,bert_size*2=hidden_size] 

        # dropout 
        tensor_outputs = self.dropout(tensor_outputs) 

        return tensor_outputs

    def entity_start_state(self,MASK_IND,sequence_output): #  sequence_output: [max_length, bert_size]
        if MASK_IND >= self.max_length:
            MASK_IND = 0
        res = sequence_output[MASK_IND]
        res = self.LayerNorm(gelu(self.tf_dense(res)))

        return res.unsqueeze(0) # [1, hidden_size]

    def tokenize_FewRel(self,inputs): #input: raw_tokens_dict 将tokens转化为ids
        tokens = inputs['tokens']
        pos_head = inputs['h'][2][0]
        pos_tail = inputs['t'][2][0]


        head_type_ = inputs['head_type']
        tail_type_ = inputs['tail_type']
        # head_type = [word.lower() if word.isupper() else word for word in head_type_]
        # tail_type = [word.lower() if word.isupper() else word for word in tail_type_]
        # head_type = head_type_#.lower()
        # tail_type = tail_type_#.lower()
        head_type0 = head_type_.split(' ')
        tail_type0 = tail_type_.split(' ')


        re_tokens,cur_pos = ['[CLS]',],0
        #获取类型
        # text = (" ").join(tokens)
        # string_ner = nlp.ner(text)
        # h_tup = string_ner[pos_head[0]:pos_head[-1] + 1]
        # head_type = list(map(lambda tup: tup[1], h_tup))
        # t_tup = string_ner[pos_tail[0]:pos_tail[-1] + 1]
        # tail_type = list(map(lambda tup: tup[1], t_tup))

        for token in tokens:
            token=token.lower() 
            if cur_pos == pos_head[0]: 
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0]:
                re_tokens.append('[unused1]')

            re_tokens+=self.tokenizer.tokenize(token)

            if cur_pos==pos_head[-1]:
                re_tokens.append('[unused2]')
                # hl = [head_type,'[unnsed4]']
                # re_tokens.extend(hl)#实体类型加入tokens
            if cur_pos==pos_tail[-1]:
                re_tokens.append('[unused3]')
                # tl = [tail_type, '[unnsed4]']
                # re_tokens.extend(tl)
            
            cur_pos+=1
        re_tokens.append('[SEP]')

        head, tail = [], []
        for cur_pos in range(pos_head[0],pos_head[-1]+1):
            head += self.tokenizer.tokenize(tokens[cur_pos])
            # head_type += self.tokenizer.tokenize(type_head[cur_pos].lower())
        for cur_pos in range(pos_tail[0],pos_tail[-1]+1):
            tail += self.tokenizer.tokenize(tokens[cur_pos])
            # tail_type += self.tokenizer.tokenize(type_tail[cur_pos].lower())
        head_type, tail_type = [], []
        for h_t in head_type0:
            head_type+=self.tokenizer.tokenize(h_t)
        for t_t in tail_type0:
            tail_type += self.tokenizer.tokenize(t_t)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        indexed_head = self.tokenizer.convert_tokens_to_ids(head)
        indexed_tail = self.tokenizer.convert_tokens_to_ids(tail)

        indexed_head_type = self.tokenizer.convert_tokens_to_ids(head_type)
        indexed_tail_type = self.tokenizer.convert_tokens_to_ids(tail_type)

        avai_len = len(indexed_tokens)

        indexed_tokens = torch.tensor(indexed_tokens).unsqueeze(0).long() 
        indexed_head = torch.tensor(indexed_head).unsqueeze(0).long() 
        indexed_tail = torch.tensor(indexed_tail).unsqueeze(0).long()

        indexed_head_type = torch.tensor(indexed_head_type).unsqueeze(0).long()
        indexed_tail_type = torch.tensor(indexed_tail_type).unsqueeze(0).long()


        if self.cuda: indexed_tokens,indexed_head,indexed_tail,indexed_head_type, indexed_tail_type= indexed_tokens.cuda(), indexed_head.cuda(), indexed_tail.cuda(),indexed_head_type.cuda(),indexed_tail_type.cuda()
        return indexed_tokens, indexed_head, indexed_tail, indexed_head_type, indexed_tail_type, avai_len #tokens_id, head_id, tail_id, text_len
