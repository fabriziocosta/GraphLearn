import math 
import networkx as nx


colordict={'black':0,'red':1,
'green':2,
'yellow':3,
'blue':4,
'cyan':6,
'magenta':5,
'gray':7}

#http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
def color(symbol,col='red'):
    return '\x1b[1;3%d;48m%s\x1b[0m' % (colordict[col],symbol)

def colorize(symbol,nodecolor,edgecolor,usecolor,colorlabel,node=None):
    if not usecolor:
        return symbol
    if node==None: 
        # we are in an edge and we need to use color
        return color(symbol,edgecolor)
    else:
        # node 
        mycolor = nodecolor if colorlabel==None else d.get('colorlabel',nodecolor)
        return color(symbol,mycolor)



def nx_to_ascii(graph,xmax=80,ymax=20,
        debug=None, 
        label='label',
        nodecolor='red',
        edgecolor='black',
        edgesymbol='.',
        usecolors=True,
        colorlabel=None):
    '''
        debug would be a path to the folder where we write the dot file.
    '''

    # ok need coordinates and a canvas
    canvas = [ list(' '*(xmax+1)) for i in range(ymax+1)]
    pos=nx.graphviz_layout(graph,prog='neato', args="-Gratio='2'")

    # alternative way to get pos. for chem graphs.
    # seems a little bit unnecessary now... maybe ill built it later
    #import molecule
    #chem=molecule.nx_to_rdkit(graph)
    #m.GetConformer().GetAtomPosition(0)


    # transform coordinates
    weird_maxx = max([x for (x,y) in pos.values()])
    weird_minx = min([x for (x,y) in pos.values()])
    weird_maxy = max([y for (x,y) in pos.values()])
    weird_miny = min([y for (x,y) in pos.values()])


    xfac= (weird_maxx - weird_minx) / xmax
    yfac= (weird_maxy - weird_miny) / ymax
    for key in pos.keys():
       wx,wy = pos[key]
       pos[key] = (int((wx - weird_minx) / xfac ),int((wy-weird_miny)/yfac))
    
    # draw nodes
    for n,d in graph.nodes(data=True):
        if 'label' in d:
            symbol=str(d['label'])
        else:
            symbol = str(n)
    
        x,y = pos[n]
        for e in symbol:
            try:
                canvas[y][x] = colorize(e,nodecolor,edgecolor,usecolors,colorlabel,node=d) # need to adress the row first
            except:
                print y,x
                return ''
            if x < xmax:
                x+=1
            else:
                continue

        
    # draw edges
    for (a,b) in graph.edges():
        ax,ay = pos[a]
        bx,by = pos[b]
        resolution =  max(3,  int(math.sqrt( (ax-bx)**2+(ay-by)**2) )) 
        dx = float((bx-ax))/resolution
        dy = float((by-ay))/resolution
        for step in range(resolution):
            x=int(ax+dx*step)
            y=int(ay+dy*step)
            if canvas[y][x] == ' ':
                canvas[y][x] = colorize(edgesymbol,nodecolor,edgecolor,usecolors,colorlabel) # need to adress the row first

    canvas = '\n'.join( [ ''.join(e) for e in canvas])
    if debug:

        path= "%s/%s.dot" %(debug, hash(graph))
        canvas+="\nwriting graph:%s" % path
        nx.write_dot(graph,path) 
    
    return canvas
   









if __name__ == "__main__":
    graph=nx.path_graph(3)
    stuff = nx_to_ascii(graph)
    print stuff
