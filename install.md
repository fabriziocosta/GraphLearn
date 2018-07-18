

## Packetmanager basics for fresh systems
Obviously not all are necessary, still leave them here in case ill have to install on a fresh system
```bash
# -y skips the confirmation question
sudo apt-get -y install python-pip git libfreetype6-dev graphviz libgraphviz-dev  liblapack-dev  liblapack3 libopenblas-base  libopenblas-dev git vim-gtk htop python-rdkit  python-tk  tmux  pandoc
sudo apt-get -y update
sudo apt-get -y upgrade
sudo init 6
```

Fish not required for graphlearn but my clone/makepath scripts are fish :D
```bash
sudo apt-add-repository ppa:fish-shell/release-2
sudo apt-get update
sudo apt-get install fish
```



##  Get all the graphlearn repositories
```bash
Use the fish script below to "getpythonpath"  and "cloneall"
```


## PIP dependencies
```bash
pip install -r EDEN/EDeN/requirements.txt  # breaks at some point..
pip install sklearn requests jupyter toolz dill scipy joblib networkx matplotlib pillow

update: numpy: 1.13.3 networkx: 1.9 scipy: 0.19.1 sklearn: 0.19.0
```



## The Fish Script

```bash
#!/usr/bin/fish
#
#######################################
#  CUTE UTILITY 
#######################################



set gl ~/GRAPHLEARN
set ed ~/EDEN

set dirlist $gl/GraphLearn $gl/GraphLearn_examples $gl/LearnedLayer \
     $gl/Generative_Adversarial_Graphlearn $gl/long_range_deps_graphlearn \
     $ed/EDeN $ed/eden_chem $ed/eden_rna $ed/eden_extra 
     

function execall --description='exec in all graphlearn git roots'
    for i in $dirlist
        cd $i
        echo "####################################"
        eval $argv  # eval :)
        echo $PWD 
        cd -
    end
end


#######################################
# Functionality  
#######################################



# setup gl/eden
function cloneall 
    mkdir GRAPHLEARN 
    cd GRAPHLEARN
    git clone git@github.com:fabriziocosta/GraphLearn
    git clone git@github.com:fabriziocosta/GraphLearn_examples
    git clone git@github.com:smautner/Generative_Adversarial_Graphlearn
    git clone git@github.com:smautner/long_range_deps_graphlearn
    git clone git@github.com:smautner/LearnedLayer
    cd ..
    mkdir EDEN
    cd EDEN
    git clone https://github.com/smautner/EDeN
    git clone git@github.com:smautner/eden_chem
    git clone git@github.com:smautner/eden_rna
    git clone https://github.com/fabriziocosta/eden_extra
    cd ..
end
    

function getpythonpath 
    echo "set -x PYTHONPATH" (string join ':' $dirlist)
end 

```
