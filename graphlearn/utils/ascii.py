import math

def nx_to_ascii(graph,xmax=80,ymax=20):
    # ok need coordinates..
    canvas = [ list(' '*(xmax+1)) for i in range(ymax+1)]
    pos=nx.graphviz_layout(graph,prog='neato')

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
            canvas[y][x] = e # need to adress the row first
            if x < xmax:
                x+=1
            else:
                continue

        
    # draw edges
    
    for (a,b) in graph.edges():
        
        ax,ay = pos[a]
        bx,by = pos[b]
        resolution = int(math.sqrt( (ax-bx)**2+(ay-by)**2) /2)
        dx = float((bx-ax))/resolution
        dy = float((by-ay))/resolution
        for step in range(resolution):
            x=int(ax+dx*step)
            y=int(ay+dy*step)
            if canvas[y][x] == ' ':
                canvas[y][x] = '.'

    

    canvas = '\n'.join( [ ''.join(e) for e in canvas])
    
    return canvas
   









if __name__ == "__main__":
    import networkx as nx
    graph=nx.path_graph(3)
    stuff = nx_to_ascii(graph)
    print stuff
