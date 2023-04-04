#从txt中读取部分数据,并重新存入文件

import json

with open('1.txt', 'r', encoding='UTF-8-sig') as data:
    with open('2.txt', 'w+', encoding='UTF-8-sig') as step1:
        for line in data:
            result = json.loads(line)
            results = result['content']
            print(results)
            step1.write(results+"\n")