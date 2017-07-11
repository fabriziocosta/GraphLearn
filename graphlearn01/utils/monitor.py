'''
during sampling, in each step information is collected.
this information is aquired and accessible through this Monitor
'''

class Monitor(object):
    '''
    will save a list of dicts, one for each sampling step.
    during a step  info(key:value) will create a dict entry
    tick(graph , id) starts a new sampling step.
    '''

    def __getitem__(self, key):
        '''
        Parameters
        ----------
        key (int)
            intex to access OoO

        Returns
        -------
        dictionary


        this allows us to conveniently accessing the list of dictionary
        '''
        return self.content[key]

    def __len__(self):
        '''
        python builtin len
        '''
        return len(self.content)

    def __init__(self, active):
        '''
        Parameters
        ----------
        active: bool
            when active, more data will be stored.


        '''
        self.active = active
        self.content = []
        self.current_dict = {}

    def tick(self, graphwrapper, ID):
        '''
        Parameters
        ----------
        graphwrapper: graphwrapper
            current graph in the sampler

        ID: int
            current sampling step

        Returns: None
        finishes the recording for one sampling step, starts a new one
        -------
        '''
        if self.active:
            self.current_dict['id'] = ID
            self.current_dict['graphwrapper'] = graphwrapper
            self.content.append(self.current_dict)
            self.current_dict = {}

    def info(self, key, val):
        '''
        Parameters
        ----------
        key: string
        val: any

        Returns: void

        creates an entry in the current dictionary,
        info (1,2)
        info (1,2)
        => will resut in a list: {1:[2,2]}
        -------

        '''
        if self.active:
            if key in self.current_dict:
                self.current_dict[key].append(val)
            else:
                self.current_dict[key] = [val]

    def _to_string(self, d):
        '''
        Parameters
        ----------
        d: dict

        Returns
        -------
        string

        "key:value\n"
        for every entry
        '''
        s = ''
        for k, v in d.items():
            s += "%s : %s\n" % (k, str(v))
        return s

    def format(self, start=-5):
        '''
        Parameters
        ----------
        start: int (-5)
        start of the returned list(see below)


        Returns
        -------
        list of pairs (accepted_graph_description, [accepted_graph, failed spawn,failed spawn, .. ]
        '''
        retlist = []
        d = self.content[0]
        current = [self._to_string(d), [d['graphwrapper'].base_graph()]]
        for d in self.content[1:]:
            if d.get('accepted:', [False])[0] == True:
                retlist.append(current)
                current = [self._to_string(d), [d['graphwrapper'].base_graph()]]
            else:
                current[1].append(d['graphwrapper'].base_graph())
        else:
            retlist.append(current)

        return retlist[start:]
