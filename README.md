# SI_project
Ze względu na słabe wyniki (prawie 200% gorsze) przy wykorzystaniu Bayesian network, wykorzystany został Bipartite graph (graf dwudzielny) w którym kolumny odpowiadają przechodniom z obecnej klatki(frame t+1), a wiersze przechodniom z poprzedniej (frame t). Na potrzeby działania programu został dopisany wiersz, ktory wskazuje na brak wystapienia przechodnia w klatce poprzedniej.
### Optymalizacja grafu
W celu optymalizacji grafu został wykorzystany Hungarian algorithm wykorzystując implementacje biblioteki [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
### Korelacja histogramu
Obliczane jest podobieństwo między histogramami obiektów na podstawie współczynnika korelacji. Dzięki dwuwymiarowym histogramom, które uwzględniają zarówno odcienie barw jak i nasycenie udało się poprawić znacznie wyniki dzięki mnniejszemu wpływowi zmiany pozycji przechodnia.
### Indeks Jaccarda
Iloraz mocy części wspólnej zbiorów i mocy sumy tych zbiorów - zwraca podobienstwo miedzy przechodniami
### Structural similarity
Funkcja oblicza wartość podobieństwa strukturalnego między dwoma obiektami. SSIM jest miarą podobieństwa strukturalnego między dwoma obrazami, uwzględniając zarówno informacje o jasności, kontraście, jak i strukturze.

### Porownanie wynikow z bboxes_gt
![Screenshot from 2024-06-13 00-20-49](https://github.com/ArturPrentki/SI_project/assets/76473377/77227bc5-333d-446f-a365-5606ba53b849)
