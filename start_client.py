import time
from tqdm import tqdm
print('start')
try:
    from bert_base.client import BertClient
except ImportError:
    raise ImportError('BertClient module is not available, it is required for serving HTTP requests.'
                      'Please use "pip install -U bert-serving-client" to install it.'
                      'If you do not want to use it as an HTTP server, '
                      'then remove "-http_port" from the command line.')

# 指定服务器的IP 127.0.0.1:49164 5555
# BertClient(ip='xxx', ner_model_dir='D:\Projects\Wunianyue\BERT-BiLSTM-CRF-NER\output', show_server_config=False, check_version=False, check_length=False, mode='NER')
with BertClient(mode='NER') as bc:
    start_t = time.perf_counter()
    # text = text.replace(' ', '-') data句子间不能有空格。
    df_path = r'data/add_data/yizhu_301_1000.txt'  # data数据最后一行要为空
    df = open(df_path, 'r+', encoding='utf-8')
    list = []
    l=[] # 要把每个字用空格分隔，放入训练?
    for line in df:
        if line!='\n':
            l.append(' '.join(line))
            list.append(line[:len(line) - 1])
    print(len(list))

    print('start')
    rst = bc.encode(l)  # 测试同时输入两个句子，多个输入同理
    k = 0
    with open("annotationdata/301_1000_BIO.txt", "w", encoding='utf-8') as f:
        for index in tqdm(range(0,len(rst))):
            try:
                f.writelines(" ".join(rst[index]))
            except:
                k = k+1
                pass

    with open("annotationdata/bert_301_1000.txt", "w", encoding='utf-8') as f:
        m = 0
        j = 0
        count = 0
        for index in tqdm(range(0, len(list))):
            if(len(list[index])!=len(rst[index])):
                    print("error in " + str(index))
                    count = count + 1
            try:
                for i in range(0, len(list[index])):
                    f.writelines(list[index][i] + ' ' + rst[index][i] + '\n')
                f.writelines('\n')
                m = m +1
            except:
                j = j + 1
                pass
    print(k)
    print(j)
    print(m)
    print(count)
    print(time.perf_counter() - start_t)
#     2.5W
