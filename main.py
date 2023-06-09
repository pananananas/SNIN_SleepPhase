import torch as t
from torch import nn
import models
import datasets
from torch.utils.data import DataLoader
import math

X = t.tensor([[1],[2],[3],[4]], dtype=t.float32)
Y = t.tensor([[2],[4],[6],[8]], dtype=t.float32)
"""X i Y - zbiór uczący. Każda podlista to jest pojedyńcza próbka, która może zawierać wiele cech, w tym przypadku zawiera tylko jedna
ilośc pod list oznacza ilosc probek.
Gdyby ilosc cech w probce wynosila 3 (np. wzrost, wiek, waga), to zapisalibysmy tensor jako [[183, 22, 75], [199, 23, 100],...]"""

X_test = t.tensor([5], dtype=t.float32)
"""X_test - zbior testowy - tutaj pojedyncza liczba, ktora podajemy na wejscie sieci"""

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features
n_epochs = 100
learning_rate = 0.01
"""n_samples ... learning_rate - atrybuty modelu, takie jak ilośc wejść, wyjsc, ilosc epok/serii etc/"""

model = models.LinearModel(input_size, output_size)
loss = nn.MSELoss()
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)
"""model - model sieci neuronowej, w naszym przypadku jest to pojedyńcza, liniowa warstwa (wiecej w models.py)
   loss - funkcja strat, tutaj sredni blad kwadratowy (MSE). Bierze wartosc jaka wyplul model, odejmuje od wartości rzeczywistej
(pobranej z tensora Y) i bierze w kwadrat, chyba. Wartość funkcji strat posłuży później do obliczenia delty wag.
   optimizer - wsm troche tajemnica. Wiem ze służy do aktualizowania wag modelu i wprowadza (o dziwo) jakąs optymalizacje"""

print(f"before training: {model(X_test).item():.5f}")

for epoch in range(n_epochs):
    y_pred = model(X)
    # przepusc przez model wartosc x ze zbioru uczacego, zapisz wynik

    l = loss(Y, y_pred)
    # oblicz funkcje strat: (y - y_pred)^2

    l.backward()
    """oblicza, w jaki sposób zmiana l zalezy od zmiany wartości wag w danej iteracji. To jest po prostu różniczka dl/dw, inaczej gradient.
    Można się zastanawiać: dla czego nie zapisujemy tego do jakiejs zmiennej, zeby pozniej te wagi zaktualizowac? Gdzie
    sie podziewa wartosc tej różniczki? Gradient ten jest właściwością (property) wektora wag, który jest ukryty w naszym modelu. 
    Metoda backward ma dostęp do tych wag przez to, że przekazaliśmy do funkcji loss model (a raczej, wynik z modelu), w którym te wagi są zawarte."""

    optimizer.step()
    optimizer.zero_grad()
    # aktualizujemy wagi oraz zerujemy gradient - nie chcemy, zeby gradient z iteracji i miał wpływ na iterację i+1

    if epoch % 10 == 0:
        [weights, bias] = model.parameters()
        print(f"epoch {epoch+1}: w = {weights[0][0]:.3f}, loss = {l:.8f}")

print(f"after training: {model(X_test).item():.5f}")
print()
print("-----Zaladowanie z pliku wierszy danych i iteracja po nich-----")

"""Nie chce mi sie specjalnie szukac csv do tego, wiec zakomentowane"""
# dataset = datasets.BasicDataset()
# batch_size = 10
# dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# n_epochs = 2
# n_samples = len(dataset)
# n_batches = math.ceil(n_samples/batch_size)

# """Z prezentacji jedna epoka to cały zbiór uczący. Tutaj jak widać nie mamy więcej niż jednego pliku z ktorego bierzemy dane.
# Domyslam się, że PT bierze sobie wszystkie dane z pliku, który wczytaliśmy i dzieli je tak, zeby się nie powtarzały między eksperymentami"""
# for epoch in range(n_epochs):
#     for i, (inputs, labels) in enumerate(dataloader):
#         # zakladam, ze PT pilnuje, zeby dane sie nie powtarzaly, bo tutaj mamy wiele epok czerpiacych z teoretycznie jednego zbioru.

#         if (i+1) % 5 == 0:
#             print(f"epoch {epoch+1}/{n_epochs}, step {i+1}/{n_batches}")


