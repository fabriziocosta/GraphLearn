


def selectdraw(graphs,chem=False,**args):
    if chem:
        import draw_openbabel as dob
        dob.draw(graphs,**args)
    else:
        import draw
        draw.graphlearn(graphs,**args)