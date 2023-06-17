import torch as t
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LinearModel, self).__init__()

        self.linear1 = nn.Linear(input_dim, output_dim) # pojedyncza warstwa liniowa

    def forward(self, x):
        return self.linear1(x)
    
"""W taki sposób w PT elegancko modyfikuje się modele sieci neuronowych. Jako główne pola tej klasy (tutaj: jedyne pola) przyjmuje się
warstwy tej sieci. Typów warstw jest wiele, różnią się między innymi rodzajem funkcji aktywacji: Step, Sigmoid, TanH, ReLU, Softmax...
Z tego co rozumiem, warstwa liniowa nn.Linear nie posiada jako-takiej funkcji aktywacji.
Warstwa liniowa bierze wejście x, mnoży przez jakiś współczynnik, i dodaje stałą. To jest po prostu funkcja liniowa, ale rozszerzona do macierzy.
Warstwy liniowe są bardzo nudne, z nich samych ciekawego modelu sie nie ulepi. Wiele warstw liniowych połączonych ze sobą daje de-facto jedną warstwę 
liniową. Dlatego też stosuje się różne kombinacje warstw:"""

class NonlinModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes) -> None:
        super(NonlinModel, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # Warstwa Relu, nieliniowa
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
    
"""Pytanie: Kiedy powinniśmy stosować jaką warstwe, skąd to mam wiedzieć?
Zależy jakie mamy zadanie, o której warstwie z kolei myślimy (pierwsza, środkowe, ostatnia...), etc.

Dla przykładu, w problemach wieloklasowych (kot, pies, kaczka, lis, okoń...) w ostatniej warstwie warto dać Softmaxa, który
"ściska" wartości wyjściowe z poprzednich warstw zamieniając je na wartości prawdopodobieństwa, które sumują sie do 1.
W warstwach ukrytych dajemy TanH - tangens hiperboliczny - lub ReLU, gdy nie wiemy jaki rodzaj warstwy dać. 
Stepa sie nie wykorzystuje w praktyce
Sigmoide wykorzystuje się albo rzadko, albo w ostatniej warstwie klasyfikacji binarej"""