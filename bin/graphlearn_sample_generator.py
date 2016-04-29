

text='''verbose=1 # sets verbose level, is not passed to sampler
model="blub.gs",
out="DONTKNOW",
graph_iter=None,
probabilistic_core_choice=True,
score_core_choice=False,
max_size_diff=-1,
similarity=-1,
n_samples=None,
proposal_probability=False,
batch_size=10,
n_jobs=0,
target_orig_cip=False,
n_steps=50,
quick_skip_orig_cip=False,
improving_threshold=-1,
improving_linear_start=0,
accept_static_penalty=0.0,
accept_min_similarity=0.0,
select_cip_max_tries=20,
burnin=0,
backtrack=0,
include_seed=False,
keep_duplicates=False,
monitor=False,
init_only=False'''

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
                        
                            
