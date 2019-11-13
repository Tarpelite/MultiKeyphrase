import json
import sys
import copy

class InputExample(object):
    """

    A single training/test example for multi-keyphrase generation 

    based on One2One pattern (1 input document cluster to 1 keyphrase pertime)

    """
    def __init__(self, guid, cluster_text, keyphrase=None):
        self.guid = guid
        self.cluster_text = cluster_text
        self.keyphrase = keyphrase