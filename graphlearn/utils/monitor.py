


class Monitor(object):
    '''
    saves a list of dicts, dicts get info(key:value) attached
    tick(graph , id) moves forward while setting graph and id
    '''

    # THINGS HERE ARE FOR COLLECTING DATA
    #
    def __getitem__(self,key):
        return self.content[key]

    def __init__(self,active):
        self.active=active
        self.content=[]
        self.current_dict={}

    def tick(self,graphwrapper,ID):
        if self.active:
            self.current_dict['id']=ID
            self.current_dict['graphwrapper']=graphwrapper
            self.content.append(self.current_dict)
            self.current_dict={}

    def info(self,key,val):
        if self.active:
            if key in self.current_dict:
                self.current_dict[key].append(val)
            else:
                self.current_dict[key]=[val]




    # THIs IS FOR SHOWING MONITORED DATA
    def show(self, start=-5 ):
        return self.compile_list()[start:]

    def record_to_string(self,d):
        s=''
        for k,v in d.items():
            s+= "%s : %s\n" % (k,str(v))
        return s

    def compile_list(self):
        retlist=[]
        d = self.content[0]
        current=[self.record_to_string(d),[d['graphwrapper'].base_graph()]]
        for d in self.content[1:]:
            if d.get('accepted:',[False])[0] == True:
                retlist.append(current)
                current=[self.record_to_string(d),[d['graphwrapper'].base_graph()]]
            else:
                current[1].append(d['graphwrapper'].base_graph())
        else:
            retlist.append(current)

        return retlist
