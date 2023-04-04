
#from string import punctuation 
import re
def removechars(rawtext, punctuation):
    data = re.sub(r"http(s)*:(\s)*", '', rawtext)#
    data = re.sub(r'[{}]'.format(punctuation),'',data)
    return data
