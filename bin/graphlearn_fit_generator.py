

text='''verbose=1 # sets verbose level, is not passed to sampler
nbit=20,
random_state=None,
vectorizer_complexity=3,
radius_list=[0, 1],#asd
thickness_list=[1, 2],
grammar=None,
min_cip_count=2,
min_interface_count=2,
input="asd.graph",
output="model.gs",
negative_input=None,
lsgg_include_negatives=False,
grammar_n_jobs=-1,
grammar_batch_size=10,
num_graphs=200 #limit number of graphs read by data sources
num_graphs_neg=200'''

print '''
import argparse
parser = argparse.ArgumentParser()
'''
tmp=[]
for line in text.split("\n"):
    hel='no help'
    hindex=line.find("#")
    if hindex != -1:
        hel=line[hindex:].strip()
        line=line[:hindex]
    deli = line.find('=')
    if deli ==- 1:
        tmp.append((line[:-1],'',hel))
    else:
        tmp.append((line[:deli],line[deli+1:-1].strip(),hel))
    

used_names=[]

def shorten(name):
    #1. try:split by underscore, use first letters
    #2. try:use longname[:3]
    #3. use longname
    l=name.split("_")
    shortname=''.join([e[0] for e in l])
    if shortname not in used_names and len(shortname)>1:
        used_names.append(shortname)
        return shortname
    shortname = name[:3]
    if shortname not in used_names:
        used_names.append(shortname)
        return shortname
    return name



def guesstype(stri):
    if len(stri)==0:
        return "str"
    if stri in ["False","True"]:
        return "bool"
    if stri == 'None':
        return 'None'
    o=ord(stri[0])
    if o == 91:
        return "list"
    if o in range(49,58)+[45]: # 45 is minus
        return "int"
    else:
        return "str"
        
for arg, value ,helpmsg in tmp:
    # so,, what we need is, long name, short name, type,help(lol), default
    longname = arg
    shortname=shorten(longname)
    typ=guesstype(value)
    default=value if value and typ != 'str' else typ
    
    # handling list
    special_mods='nargs="+",' if typ=='list' else ''
    if typ=='list':
        typ='int'
        
    print '''parser.add_argument(
            "--%s",
            "--%s",%s
            dest="%s",
            type=%s,
            help="%s",
            default=%s) 
    '''% (shortname, longname,special_mods, longname, typ, helpmsg,default)
                        
                            
