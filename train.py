# -*- coding: utf-8 -*-
import torch
from torch import autograd, nn
from torch.nn import functional as F
from transformers import AdamW,get_linear_schedule_with_warmup
from dataloader import FewshotDataset,FewshotDataset_test
import sys
import json
from model import knowMP

def fast_tuning(W,support,support_label,query, net,steps,task_lr,N,K):
    '''
       W:               label word embedding matrix                             [N, hidden_size] 
       support:         support instance hidden states at [MASK] place          [N*K, hidden_size]    
       support_label:   support instance label id:                              [N*K]
       query:           query instance hidden states at [MASK] place            [total_Q, hidden_size]
       steps：          fast-tuning steps                                       
       task_lr:         fast-tuning learning rate for task-adaptation          
    '''
    prototype_label = torch.tensor( [i for i in range(N)]).cuda() # [0,1,2,...N]
    
    hidden_size = support.size(-1)
    fc = nn.Linear(hidden_size, hidden_size, bias=True).cuda()
    query_for = query.view(1, -1, hidden_size)
    query_for_att = fc(query_for)
    Q = query_for.size(1)
    z = torch.zeros([N, Q, K, hidden_size]).cuda()
    support_for = support.view(N, K, hidden_size)
    support_for_att = fc(support_for)
    for i in range(query_for_att.size(0)):
        for j in range(query_for_att.size(1)):
            a = support_for_att[i,:,:]
            b = query_for_att[i,j,:]
            b = b.expand(K, query_for_att.size(-1))
            z[i, j, :, :] = b * a
    # ins = z.sum(dim=-1).sum(1)
    # aa = F.softmax(z.sum(dim=-1), dim = 2).sum(1)
    ins_att = F.softmax(torch.tanh(F.softmax(z.sum(dim=-1), dim = 2).sum(1)), dim=1)
    ins_att_score = ins_att.reshape(-1,1).expand(-1,hidden_size)
    support = ins_att_score * support


    #实例级注意力机制  attention score calc
    idx = torch.zeros(N*K).long().cuda()
    for i in range(N): idx[i*K:(i+1)*K] = i # [0,0,...0,1,1...1,...N-1...N-1]
    att=(support * W[idx]).sum(-1).reshape(N,K) # ([N*K,bert_size]·[N*K,bert_size]).sum(-1) = [N*K] ——>  [N,K]
    T = 3   #温度系数
    att = F.softmax(att/T,-1).detach() # [N,K]
    # att: attention scores α_i^j

    # att=None
    for _ in range(steps):
        logits_for_instances, logits_for_classes = net(support,W) # [N*K, N], [N, N]
        if att is None:
            loss_s2v = net.loss(logits_for_instances, support_label)
            loss_v2s = net.loss(logits_for_classes, prototype_label)

            loss = loss_s2v + loss_v2s  #L_att

            grads = autograd.grad(loss,W)   #快速调优
            W = W - task_lr*grads[0]   #输出任务适应的标签词嵌入W
        else:
            Att = att.flatten() # [N*K]
            loss = torch.FloatTensor([0.0] * (N*K)).cuda()
            for i in range(N*K):
                loss[i] = net.loss(logits_for_instances[i].unsqueeze(0),support_label[i])/N
            loss_tot = Att.dot(loss)
            grad = autograd.grad(loss_tot,W)    #快速调优
            W = W - task_lr*grad[0]  #输出任务适应的标签词嵌入W

    logits_q = net(query, W)[0] # 查询推理[total_Q, n_way]
    return logits_q

def train_one_batch(idx,class_names,support0,support_label,query0,query_label,Q,net,steps,task_lr):
    '''
    idx:                batch index         
    class_names：       N categories names (or name id)             List[class_name * N]
    support0:           raw support texts                           List[{tokens:[],h:[],t:[]} * (N*K)]
    support_label:      support instance labels                     [N*K]
    query0:             raw query texts                             List[{tokens:[],h:[],t:[]} * total_Q]
    query_label:        query instance labels                       [total_Q]
    net:                PBML model
    steps：             fast-tuning steps                                       
    task_lr:            fast-tuning learning rate for task-adaptation
    '''
    N, K = net.n_way, net.k_shot

    support, query = net.coder(support0), net.coder(query0) # [N*K,bert_size]
    candidate_word_embeddings =net.get_info(class_names) # [N * [candidate word embeddings]]

    net.W[idx] = net.prework(candidate_word_embeddings) #标签词初始化

    logits_q = fast_tuning(net.W[idx],support,support_label,query,net,steps,task_lr,N,K)    #查询推理

    return net.loss(logits_q, query_label),   net.accuracy(logits_q, query_label)


def eval_model(data_loader,model,Q,val_iter,steps,task_lr):
    accs=0.0
    model.eval()

    for it in range(val_iter):
        net = model
        class_name,support,support_label,query,query_label = data_loader[0]
        loss,right = train_one_batch(0,class_name, support, support_label,query,query_label,Q,net,steps,task_lr)
        accs += right
        # logger = getLogger()
        # logger.info('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it+1)) + '\r')  # 打印训练日志

        sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it+1)) + '\r')
        sys.stdout.flush()

    return accs/val_iter


def train_model(model:knowMP, B,N,K,Q,data_dir,
            meta_lr=5e-5, 
            task_lr=1e-2,
            weight_decay = 1e-2,
            train_iter=2000,
            val_iter=2000,
            val_step=50,
            steps=30,
            save_ckpt = None,
            load_ckpt = None,
            best_acc = 0.0,
            fp16 = False,
            # fp16 = True,
            warmup_step = 200):

    n_way_k_shot = str(N) + '-way-' + str(K) + '-shot'
    print('Start training ' + n_way_k_shot)
    cuda = torch.cuda.is_available()
    if cuda: model = model.cuda()

    if load_ckpt:
        state_dict = torch.load(load_ckpt)['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('ignore {}'.format(name))
                continue
            print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)
    
    
    data_loader={}#数据加载
    data_loader['train'] = FewshotDataset(data_dir['train'],N,K,Q,data_dir['noise_rate']) 
    data_loader['val'] = FewshotDataset(data_dir['val'],N,K,Q,data_dir['noise_rate'])
    # data_loader['test'] = FewshotDataset(data_dir['test'],N,K,Q,data_dir['noise_rate'])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    coder_named_params = list(model.coder.named_parameters())

    for name, param in coder_named_params:
        if name in {'bert_ebd.word_embeddings.weight','bert_ebd.position_embeddings.weight','bert_ebd.token_type_embeddings.weight'}:
            param.requires_grad = False
            pass


    optim_params=[{'params':[p for n, p in coder_named_params 
                    if not any(nd in n for nd in no_decay)],'lr':meta_lr,'weight_decay': weight_decay},
                  {'params': [p for n, p in coder_named_params 
                    if any(nd in n for nd in no_decay)],'lr':meta_lr, 'weight_decay': 0.0},
                ]
       

    meta_optimizer=AdamW(optim_params)
    scheduler = get_linear_schedule_with_warmup(meta_optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)#创建学习率调度器

    if fp16:
        from apex import amp
        model, meta_optimizer = amp.initialize(model, meta_optimizer, opt_level='O1')

    iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0

    model.train()

    for it in range(train_iter):
        meta_loss, meta_right = 0.0, 0.0

        for batch in range(B):
            class_name, support, support_label, query, query_label = data_loader['train'][0]
            loss, right =train_one_batch(batch,class_name,support,support_label,query,query_label,Q,model,steps,task_lr)
            
            meta_loss += loss
            meta_right += right
        
        meta_loss /= B
        meta_right /= B

        meta_optimizer.zero_grad()#在每轮训练中，清空梯度，计算模型输出和损失
        if fp16:
            with amp.scale_loss(meta_loss, meta_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            meta_loss.backward()#计算梯度
        meta_optimizer.step()#更新模型参数
        scheduler.step()#更新学习率
    
        iter_loss += meta_loss
        iter_right += meta_right
        iter_sample += 1

        # logger = getLogger()
        # logger.info('step:{0:4} | \tloss:{1:2.6f}\taccuracy:{2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample)+'\r')  # 打印训练日志

        sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
        sys.stdout.flush()

        if (it+1)%val_step==0:
            print("")
            iter_loss, iter_right, iter_sample = 0.0,0.0,0.0
            acc = eval_model(data_loader['val'], model, Q, val_iter, steps,task_lr)
            print("")
            model.train()
            if acc > best_acc:
                print('Best checkpoint!')
                torch.save({'state_dict': model.state_dict()}, save_ckpt)

                best_acc = acc
    print("\n####################\n")
    print('Finish training model! Best acc: '+str(best_acc))


def evaluation(model,N,K,Q,eval_iter=10000,steps=30,task_lr=1e-2, noise_rate = 2,file_name=None,load_ckpt = None):
    if torch.cuda.is_available(): model = model.cuda()

    if load_ckpt:
        state_dict = torch.load(load_ckpt)['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # print('ignore {}'.format(name))
                continue
            # print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)

    accs=0.0
    model.eval()
    data_loader = FewshotDataset(file_name,N,K,Q,noise_rate)
    tot = {}
    neg = {}
    for it in range(eval_iter):
        net = model
        class_name,support,support_label,query,query_label = data_loader[0]
        _,right = train_one_batch(0,class_name, support, support_label,query,query_label,Q,net,steps,task_lr)
        accs += right 
        for i in class_name:
            if i not in tot:
                tot[i]=1
            else:
                tot[i]+=1
        if right <1:
            for i in class_name:
                if i not in neg:
                    neg[i]=1
                else:
                    neg[i]+=1
        # logger = getLogger()
        # logger.info('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it + 1)) + '\r')  # 打印训练日志
        sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it+1)) + '\r')
        sys.stdout.flush()
    print("")
    print(tot)
    print(neg)
    print("")

    return accs/eval_iter

#result on test set
def test(idx,class_names,support0,support_label,query0,net,steps,task_lr):
    '''
    idx:                batch index
    class_names：       N categories names (or name id)             List[class_name * N]
    support0:           raw support texts                           List[{tokens:[],h:[],t:[]} * (N*K)]
    support_label:      support instance labels                     [N*K]
    query0:             raw query texts                             List[{tokens:[],h:[],t:[]} * total_Q]
    query_label:        query instance labels                       [total_Q]
    net:                PBML model
    steps：             fast-tuning steps
    task_lr:            fast-tuning learning rate for task-adaptation
    '''
    N, K = net.n_way, net.k_shot
    support, query = net.coder(support0), net.coder(query0) # [N*K,bert_size]
    candidate_word_embeddings =net.get_info(class_names) # [N * [candidate word embeddings]]

    net.W[idx] = net.prework(candidate_word_embeddings)

    logits_q = fast_tuning(net.W[idx],support,support_label,query,net,steps,task_lr,N,K)

    _, pred = torch.max(logits_q, 1)
    return pred

def test_model(model,N,K,Q,eval_iter=10000,steps=30,task_lr=1e-2, noise_rate = 0,file_name=None,load_ckpt = None):
    if torch.cuda.is_available(): model = model.cuda()

    if load_ckpt:
        state_dict = torch.load(load_ckpt)['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # print('ignore {}'.format(name))
                continue
            # print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)

    model.eval()
    data_loader = FewshotDataset_test(file_name,N,K,Q,noise_rate)

    predict_label = []

    for it in range(eval_iter):
        net = model
        # class_name,support,support_label,query,query_label = data_loader[0]
        class_name, support, support_label, query = data_loader[it]
        # class_name = data["relation"]
        predict_query_label = test(0,class_name, support, support_label,query,net,steps,task_lr)
        predict_label.append(predict_query_label.item())

        sys.stdout.write('step: {0:4} |'.format(it + 1)+'\r')
        sys.stdout.flush()
    with open('FewRel2/pred-{}-{}.json'.format(N,K), 'w') as f:
        json.dump(predict_label, f)
    return predict_label