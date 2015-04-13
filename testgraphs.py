# ok we copy paste stuff vom beispiel...
def rfam_uri(family_id):
    return 'http://rfam.xfam.org/family/%s/alignment?acc=%s&format=fastau&download=0'%(family_id,family_id)
#def rfam_uri(family_id):
#    return '%s.fa'%(family_id)
    
rfam_id = 'RF02275' 
    
    
import re
from eden import util

def fasta_to_fasta( input ):
    for m in re.finditer(r"^(>[^\n]+)\n+([^>]+)",'\n'.join(util.read( input )), re.MULTILINE):
        if m: 
            header, seq = m.groups()
            seq = re.sub('\n','',seq)
            yield header
            yield seq
            
            
            
iterable = fasta_to_fasta(rfam_uri(rfam_id))
[line for line in iterable]


import networkx as nx
def sequence_dotbracket_to_graph(seq_info, seq_struct):
    G = nx.Graph()
    lifo = list()
    for i,(c,b) in enumerate( zip(seq_info, seq_struct) ):
        G.add_node(i, label = c) 
        if i > 0:
            #add backbone edges
            G.add_edge(i,i-1, label='-')
        if b == '(':
            lifo.append(i)
        if b == ')':
            #when a closing bracket is found, add a basepair edge with the corresponding opening bracket 
            j = lifo.pop()
            G.add_edge(i,j, label='=')
    return G


import subprocess as sp
def pre_process(input):
    lines =  fasta_to_fasta(input)


    for line in lines:
        #get a header+sequence
        header = line
        seq = lines.next()
        
        #invoke RNAfold
        cmd = 'echo "%s" | RNAfold --noPS' % seq
        out = sp.check_output(cmd, shell = True)
        #parse the output
        text = out.strip().split('\n')
        seq_info = text[0]
        seq_struct = text[1].split()[0]
        #make a graph
        G = sequence_dotbracket_to_graph(seq_info, seq_struct)
        G.graph['id'] = header
        yield G

def get_graphs():
    return pre_process(rfam_uri(rfam_id))
