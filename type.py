import time
import requests
import json
import os
def get_type(entity_qid):
    query = """
    SELECT ?entity ?entityLabel ?entityTypeLabel WHERE {
      BIND(wd:%s AS ?entity)  # 使用输入的实体QID作为变量
      ?entity wdt:P31 ?entityType.
      ?entity rdfs:label ?entityLabel.
      FILTER(LANG(?entityLabel) = "en").
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """ % entity_qid

    # 构建请求参数
    params = {
        'query': query,
        'format': 'json'
    }

    # 发送请求并获取响应
    response = requests.get('https://query.wikidata.org/sparql', params=params)
    data = response.json()

    # 处理查询结果
    result = data['results']['bindings']


    if result:
        entity_label = result[0]['entityLabel']['value']
        entity_type_label = result[0]['entityTypeLabel']['value']
        print(f"Entity: {entity_label}, Type: {entity_type_label}")
        return entity_type_label
    else:
        # 如果没有结果，尝试获取重定向实体的类型信息
        redirect_query = """
                SELECT ?targetTypeLabel WHERE {
                  VALUES ?redirectEntity { wd:%s }
                  ?redirectEntity owl:sameAs ?targetEntity.
                  ?targetEntity wdt:P31 ?targetType.
                  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                }
                """ % entity_qid

        # 构建请求参数
        redirect_params = {
            'query': redirect_query,
            'format': 'json'
        }

        # 发送请求并获取重定向实体的类型信息
        redirect_response = requests.get('https://query.wikidata.org/sparql', params=redirect_params)
        redirect_data = redirect_response.json()

        redirect_results = redirect_data['results']['bindings']
        if redirect_results:
            redirect_result = redirect_results[0]
            target_type_label = redirect_result['targetTypeLabel']['value']
            print(f"Redirected Entity: {entity_qid}")
            print(f"Target Entity Type: {target_type_label}")
            return target_type_label
        else:
            print(f"No information available for entity: {entity_qid}")

# get_type('Q16893977')

dir = 'data/FewRel/val.json'
data = json.load(open(dir, 'r', encoding='utf-8'))
# v = list(data.values())[1]
# k = list(data.keys())[1]
# for i in range(len(v)):
#     h_id = v[i]["h"][1]
#     t_id = v[i]["t"][1]
#     h_type = get_type(h_id)
#     t_type = get_type(t_id)
#     data[k][i]["head_type"] = h_type
#     data[k][i]["tail_type"] = t_type
#     time.sleep(2)
#     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# json.dump(data, (open('data/FewRel/wikidata/train.json', 'w')))
for index, (k, v) in enumerate(data.items()):
    print(k+"%%%%%%%%%%%"+str(index))
    # if index>=13:
    dict ={}
    for i in range(len(v)):
            h_id = v[i]["h"][1]
            t_id = v[i]["t"][1]
            h_type = get_type(h_id)
            t_type = get_type(t_id)
            data[k][i]["head_type"] = h_type
            data[k][i]["tail_type"] = t_type

            time.sleep(3)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+"######### "+str(index)+"～～"+str(i))
            if (i+1)%50==0:
                time.sleep(60)
    dict[k] = data[k]

    if os.path.getsize('data/FewRel/wikidata/val.json') == 0:
            json.dump(dict, (open('data/FewRel/wikidata/val.json', 'w')))
    else:
            with open('data/FewRel/wikidata/val.json', 'r+') as file:
                new_data = json.load(file)
                new_data.update(dict)
                file.seek(0)  # 将文件指针移回文件开头
                json.dump(new_data, file)
    time.sleep(300)
# json.dump(data, (open('data/FewRel/wikidata/val.json', 'w')))


