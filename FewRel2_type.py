import os
import json
import requests
import ssl
from requests.adapters import HTTPAdapter
import time

# 自定义HTTPS适配器以支持TLSv1.2及以上
class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers('HIGH:!DH:!aNULL')  # 设置强加密
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # 禁用TLSv1和TLSv1.1
        kwargs['ssl_context'] = context
        super(TLSAdapter, self).init_poolmanager(*args, **kwargs)

# 创建会话并挂载自定义适配器
session = requests.Session()
session.mount("https://", TLSAdapter())

# API基本信息
API_KEY = "YOUR_API_KEY"
AUTH_ENDPOINT = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
SERVICE = "http://umlsks.nlm.nih.gov"
UMLS_ENDPOINT = "https://uts-ws.nlm.nih.gov/rest"

# 需要尝试的UMLS API版本列表
VERSIONS = ["2020AA", "2019AB", "2018AA", "2018AB"]

# 缓存字典，用于存储已查询的CUI语义类型
semantic_type_cache = {}

# 获取TGT
def get_tgt(api_key):
    data = {"apikey": api_key}
    response = session.post(AUTH_ENDPOINT, data=data)
    response.raise_for_status()
    return response.headers["location"]

# 使用TGT获取ST
def get_service_ticket(tgt):
    data = {"service": SERVICE}
    response = session.post(tgt, data=data)
    response.raise_for_status()
    return response.text

# 获取CUI的语义类型
def get_semantic_types(cui, service_ticket, version):
    response = session.get(f"{UMLS_ENDPOINT}/content/{version}/CUI/{cui}", params={"ticket": service_ticket})
    response.raise_for_status()
    data = response.json()
    semantic_types = [st["name"] for st in data.get("result", {}).get("semanticTypes", [])]
    return semantic_types

# 获取未缓存的CUI的语义类型并更新缓存
def batch_get_semantic_types(cui_list, tgt):
    for cui in cui_list:
        # 跳过已缓存的CUI
        if cui in semantic_type_cache:
            continue
        for version in VERSIONS:
            try:
                service_ticket = get_service_ticket(tgt)  # 获取服务票据
                semantic_types = get_semantic_types(cui, service_ticket, version)
                if semantic_types:  # 如果查询到结果，缓存并返回第一个语义类型
                    semantic_type_cache[cui] = semantic_types[0]
                    break
            except requests.HTTPError as e:
                print(f"在版本 {version} 中查询CUI {cui} 语义类型时出错: {e}")
                continue
        # 若未查询到结果，缓存None
        if cui not in semantic_type_cache:
            semantic_type_cache[cui] = None

# 主函数 - 处理JSON数据并获取语义类型
def process_data(file_path, output_path, api_key):
    json_data = json.load(open(file_path, 'r', encoding='utf-8'))
    tgt = get_tgt(api_key)  # 获取TGT

    for k, entry in enumerate(json_data):
        print(f"Processing entry {k}")
        support = entry['meta_train']
        query = entry['meta_test']

        # 收集所有需要查询的CUI
        cui_list = {query["h"][1], query["t"][1]}
        for support_list in support:
            for sample in support_list:
                cui_list.add(sample["h"][1])
                cui_list.add(sample["t"][1])

        # 批量查询未缓存的CUI
        batch_get_semantic_types(cui_list, tgt)

        # 填充查询结果
        entry['meta_test']["head_type"] = semantic_type_cache.get(query["h"][1], "")
        entry['meta_test']["tail_type"] = semantic_type_cache.get(query["t"][1], "")
        for support_list in support:
            for sample in support_list:
                sample["head_type"] = semantic_type_cache.get(sample["h"][1], "")
                sample["tail_type"] = semantic_type_cache.get(sample["t"][1], "")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 写入结果到文件
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, indent=4)
    print("All entries processed and saved.")

# 执行处理
process_data('FewRel2/test-10-5.json', 'FewRel2/test-10-5_type.json', API_KEY)
