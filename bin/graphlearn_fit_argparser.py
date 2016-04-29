
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
            "--ver",
            "--verbose",
            dest="verbose",
            type=int,
            help="# sets verbose level, is not passed to sampler",
            default=1) 
    
parser.add_argument(
            "--nbi",
            "--nbit",
            dest="nbit",
            type=int,
            help="no help",
            default=20) 
    
parser.add_argument(
            "--rs",
            "--random_state",
            dest="random_state",
            type=None,
            help="no help",
            default=None) 
    
parser.add_argument(
            "--vc",
            "--vectorizer_complexity",
            dest="vectorizer_complexity",
            type=int,
            help="no help",
            default=3) 
    
parser.add_argument(
            "--rl",
            "--radius_list",nargs="+",
            dest="radius_list",
            type=int,
            help="#asd",
            default=[0, 1]) 
    
parser.add_argument(
            "--tl",
            "--thickness_list",nargs="+",
            dest="thickness_list",
            type=int,
            help="no help",
            default=[1, 2]) 
    
parser.add_argument(
            "--gra",
            "--grammar",
            dest="grammar",
            type=None,
            help="no help",
            default=None) 
    
parser.add_argument(
            "--mcc",
            "--min_cip_count",
            dest="min_cip_count",
            type=int,
            help="no help",
            default=2) 
    
parser.add_argument(
            "--mic",
            "--min_interface_count",
            dest="min_interface_count",
            type=int,
            help="no help",
            default=2) 
    
parser.add_argument(
            "--inp",
            "--input",
            dest="input",
            type=str,
            help="no help",
            default=str) 
    
parser.add_argument(
            "--out",
            "--output",
            dest="output",
            type=str,
            help="no help",
            default=str) 
    
parser.add_argument(
            "--ni",
            "--negative_input",
            dest="negative_input",
            type=None,
            help="no help",
            default=None) 
    
parser.add_argument(
            "--lin",
            "--lsgg_include_negatives",
            dest="lsgg_include_negatives",
            type=bool,
            help="no help",
            default=False) 
    
parser.add_argument(
            "--gnj",
            "--grammar_n_jobs",
            dest="grammar_n_jobs",
            type=int,
            help="no help",
            default=-1) 
    
parser.add_argument(
            "--gbs",
            "--grammar_batch_size",
            dest="grammar_batch_size",
            type=int,
            help="no help",
            default=10) 
    
parser.add_argument(
            "--ng",
            "--num_graphs",
            dest="num_graphs",
            type=int,
            help="#limit number of graphs read by data sources",
            default=200) 
    
parser.add_argument(
            "--ngn",
            "--num_graphs_neg",
            dest="num_graphs_neg",
            type=int,
            help="no help",
            default=20) 
    
