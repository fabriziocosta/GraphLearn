


def selectdraw(graphs,chem=False,**args):
    if chem:
        import draw_openbabel as dob
        dob.draw(graphs,**args)
    else:
        import draw
        draw.graphlearn(graphs,**args)




# this is copy pasted in EVERY notebook. now its here 
import matplotlib.pyplot as plt
def plot_scores(scoreslist='list of lists', labels=[]):

    if len(labels)==0:
        labels= ['index: %d' % i for i in len(scoreslist)]

    plt.figure(figsize=(10,5))

    for j,scores in enumerate(scoreslist):
        plt.plot(scores, label=labels[j])

    maa= max([ max(l) for l in scoreslist  ])
    mii= min([min(l) for l in scoreslist])
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim(mii*1.1,maa*1.1)
    plt.show()


# i might need this one day
def chunk(l,n):
    return [l[i:i+n] for i in xrange(0, len(l), n)]