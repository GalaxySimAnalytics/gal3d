from collections.abc import Iterable
import time
import logging

import numpy as np
from tqdm import tqdm

from .point import Particles
from .field import SphField

from .shape import Structure3D
from .optimization.optimizer import Optimizer
from .optimization.result import ModelResult



class Gal3DExecutor:
    
    
    def __init__(self, pos, mass, config = None):
        
        self.pos = pos
        self.mass = mass
        
        
    
        
        