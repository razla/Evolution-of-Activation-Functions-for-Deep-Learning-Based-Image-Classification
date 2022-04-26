from cartesian.cgp import *
import torch.nn as nn
import torch

primitives = [
    Primitive("max", torch.max, 2),
    Primitive("min", torch.min, 2),
    Primitive("add", torch.add, 2),
    Primitive("sub", torch.sub, 2),
    Primitive("mul", torch.mul, 2),
    Primitive("tanh", torch.tanh, 1),
    Primitive("ReLU", nn.ReLU(), 1),
    Primitive("LeakyRelu", nn.LeakyReLU(), 1),
    Primitive("ELU", nn.ELU(), 1),
    Primitive("Hardshrink", nn.Hardshrink(), 1),
    Primitive("CELU", nn.CELU(), 1),
    Primitive("Hardtanh", nn.Hardtanh(), 1),
    Primitive("Hardswish", nn.Hardswish(), 1),
    Primitive("Softshrink", nn.Softshrink(), 1),
    Primitive("RReLU", nn.RReLU(), 1),
    Symbol("x_0"),
]

pset = PrimitiveSet.create(primitives)
