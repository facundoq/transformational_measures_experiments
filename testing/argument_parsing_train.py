


from experiment import model_loading, training
import argparse,argcomplete
from typing import Tuple

def list_parser(s:str):
    s=s.strip()
    if s=="":
        return []
    delim=","
    string_list=s.split(delim)
    return [int(n) for n in string_list]

def parse_args():

    bool_parser=lambda x: (str(x).lower() in ['true','1', 'yes'])


    parser = argparse.ArgumentParser(description="Script to train a models with a dataset and transformations")

    parser.add_argument('-epochs', metavar='epo'
                        , help=f'Epochs to train the model'
                        ,required=True
                        , type=int)


    parser.add_argument('-model', metavar='m',
                        help=f'Model to train/use. Allowed values: '
                        ,required=True)


    parser.add_argument('-dataset', metavar='d',
                        help=f'Dataset to train/eval models. Allowed values: ", "'
                        ,required=True)

    #argcomplete.autocomplete(parser)

    args = parser.parse_args()

    return args

import sys

if __name__ == "__main__":
    print(sys.argv)
    args = parse_args()
