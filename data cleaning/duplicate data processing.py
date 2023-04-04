import shutil
readPath='数据1.txt'
writePath='数据2.txt'
lines_seen=set()
outfiile=open(writePath,'a+',encoding='utf-8')
f=open(readPath,'r',encoding='utf-8')
for line in f:
    if line not in lines_seen:
        outfiile.write(line)
        lines_seen.add(line)
