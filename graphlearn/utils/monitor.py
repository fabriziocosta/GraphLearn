


class Monitor(object):
    '''
    saves a list of dicts, dicts get info(key:value) attached
    tick(graph , id) moves forward while setting graph and id
    '''

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
