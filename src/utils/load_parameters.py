from pathlib import Path
import yaml
import os
import sys

def load_parameters():
    path = Path('.').parent.absolute()
    path = os.path.join(path, 'parameters.yaml')
    with open(path) as file:
        doc = yaml.full_load(file)

    # Model parameters

    if(doc['model'][0]['type'] is None):
        sys.exit("Type not specified: \n" +
            "- Sequential \n" +
            "- Model"
        )

    if((doc['model'][1]['in_nodes'] is None) or
        (doc['model'][2]['out_nodes']) is None):
        sys.exit("in OR out nodes not specified")

    if((doc['model'][3]['layer_num'] is None) or
        (doc['model'][4]['layer_nodes'] is None)):
        sys.exit("layer_num OR layer_nodes not specified")

    # fit parameters

    if(doc['fit'][0]['loss_func'] is None):
        sys.exit("Loss Function not specified")

    return doc['model'], doc['fit']
