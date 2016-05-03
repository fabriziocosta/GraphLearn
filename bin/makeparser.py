import argparse


def makeparser(text):
    # so the optionlist, has equal signs and sometimes '#' letters.
    # -> split according to those
    tmp = []
    for line in text.split("\n"):
        hel = 'no help'
        hindex = line.find("#")
        if hindex != -1:
            hel = line[hindex:].strip()
            line = line[:hindex]
        deli = line.find('=')
        if deli == - 1:
            tmp.append((line[:-1], '', hel))
        else:
            tmp.append((line[:deli], line[deli + 1:-1].strip(), hel))
    used_names = []

    # optionlist gives a long name for the parameters
    # short names are nicer for users
    # we try to guess a short name here.
    def shorten(name):
        # 1. try:split by underscore, use first letters
        # 2. try:use longname[:3]
        # 3. use longname
        l = name.split("_")
        shortname = ''.join([e[0] for e in l])
        if shortname not in used_names and len(shortname) > 1:
            used_names.append(shortname)
            return shortname
        shortname = name[:3]
        if shortname not in used_names:
            used_names.append(shortname)
        return shortname
        return name

    # making a parser...

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for arg, value, helpmsg in tmp:
        # so,, what we need is, long name, short name, type,help(lol), default

        longname = arg
        shortname = shorten(longname)
        value = eval(value)
        typ = type(value)
        default = value
        # handling list: lists are handled weirdly in argparse,
        # so here is another exception :)
        nargs = "+" if typ == list else None
        if typ == list:
            typ = int

        # print arg,default,type(default),default
        parser.add_argument(
            "--" + shortname,
            "--" + longname,
            nargs=nargs,
            dest=longname,
            type=typ,
            help=helpmsg,
            default=default)
    return parser