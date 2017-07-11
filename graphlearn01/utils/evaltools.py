'''
this runs RNAshapes
dotbracket string -> higher level dotbracket shape indication string
'''

import subprocess as sp
def dotbracket_to_shape(dbstring, shapesversion=2):
    if shapesversion == 3:
        cmd = 'RNAshapes --mode abstract --shapeLevel 3 "%s"' % dbstring
    else:
        cmd = 'RNAshapes -D "%s" -t 3 ' % dbstring
    out = sp.check_output(cmd, shell=True)
    return out.strip()


def test():
    db = "(((...((...)))))"
    if dotbracket_to_shape(db) != "[[]]":
        print "dotbracket failed"
