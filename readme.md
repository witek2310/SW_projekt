# PROJEKT SW

Cały proces znajdywania rejestracji składa się z 2 kroków:\
-znalezienie rejestracji\
-dopasowanie najlepszych liter

## znajdowanie rejestracji

Najpierw obraz jest skalowany do mniejszego oraz zmieniany jest obraz w skali szarośći\
nastepnie znajdowanie są wszystkie kontury\
Po znalezieniu konturów, są one przybliżane do prostokątów i jeżeli nie jest to możliwe, kontur jest odrzucany\
następnie jeżeli kontur zajmuje przynajmnie 1/3 obnrazu i jest wystarczająco duży, to jest przyjmowany jako tablica

## znajdowanie liter

Ta część dla mnie była dużo trudniejsza. Próbowałem różnych sposobów, ale większość nie działała :(.\
deskryptory oraz matchShapes nie dawała satyfakcjonujących rezulatów.\
Skończyło się na maskach binarnych.\
Dla każdej z liter z rejestacji wyznaczany jest jej bounding box (na podstawie kontur) oraz obliczana jest maska binarna miesząca się w nim.\
Następnie dla każdej z liter na podstawie czcionki wyznaczany jest bounding box i skalowany jest do rozmiaru tego z rejestracji i wyznaczana jest maska.
Na podstawie tych masek wyznaczny jest ich AND oraz XOR oraz zliczana jest liczba "1".\
W idealnym przypadku AND będzie pokrywał się z maską, a XOR będzie miał same 0.\
Więc dlaczego potrzebny jest XOR?\
Problem pojawia się w przypadku "I" jako że jej biała część zajmuję duży obszar, iloczyn maski czcionki daje duży wynik (sumę "1") dla każdej z liter z rejestracji :(\
problem też pojawi się w przypadku F na rejestracji i "E" na podstawie czcionki. "E" może dać lepszy wynik.\
I tutaj przychodzi na pomoc XOR, który zlicza elemeny które się różnią. I tak w wspomnianym przypadku F na rejestracji oraz E z czcionki otrzymamy liczbę pikseli z dolnej nóżki.\
Ostatecznie obliczana jest różnica: sum(AND) - sum(XOR) i znajdowana jest litera, która maksymalizuje ten wynik i dodwana jest do listy wraz z najbardziej wysuniętym na lewo punktem litery z rejestracji\

Po przejścu przez wszystkie litery znajdujące się na rejestarcji sortowane są one po lewym punkcie i łączone w stringa.

Do poprawy jakości działania liczone są "dziury" w literach z rejestracji (funkcja holes) i na tej podstawie odpowiednie litery są brane pod uwagę.
