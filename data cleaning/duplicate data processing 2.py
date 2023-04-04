
# -*- coding: UTF-8 -*-
#程序功能是为了完成判断文件中是否有重复句子
#并将重复句子打印出来
 
res_list = []
#f = open('F:/master/master-work/code_of_graduate/LTP_data/raw_plain.txt','r')
f = open('H:/硕士-毕业论文/2019-数据清洗/数据1.txt','r',encoding='UTF-8')
res_dup = []
index = 0
file_dul = open('H:/硕士-毕业论文/2019-数据清洗/数据2.txt', 'w',encoding='UTF-8')
for line in f.readlines():
    index = index + 1
    if line in res_list:
        temp_str = ""
        temp_str = temp_str + str(index)                    #要变为str才行
        temp_line = ''.join(line)
        temp_str = temp_str+temp_line
        #最终要变为str类型
        file_dul.write(temp_str);                           #将重复的存入到文件中
    else:
        res_list.append(line)

