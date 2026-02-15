# Aplikacja Webowa do Klasyfikacji ML

> Projekt - interaktywna aplikacja webowa do klasyfikacji opartej na regułach z asystentem AI w formie czatu.

https://github.com/pawel-li/merito2
---

## Opis

Aplikacja umożliwia użytkownikowi wgranie zbioru danych CSV, wytrenowanie klasyfikatora opartego na drzewie decyzyjnym, przegląd wygenerowanych reguł klasyfikacji oraz interakcję z asystentem AI, który potrafi dodawać, usuwać i przeliczać reguły w czasie rzeczywistym. Wszystkie zmiany są natychmiastowo odzwierciedlane w tabeli reguł i macierzy pomyłek.

## Demo

https://github.com/user-attachments/assets/2fdb0b3a-6c5d-4f6a-8e3e-1e1f6e3e4d2b

### Główne funkcjonalności

- **Wgrywanie CSV** — przeciągnij i upuść lub wybierz plik CSV, aby utworzyć nowy projekt
- **Trenowanie modelu** — wybierz kolumnę docelową i wytrenuj klasyfikator Decision Tree (scikit-learn)
- **Tabela reguł** — podgląd reguł IF…THEN z metrykami per reguła (pokrycie, precyzja, jakość, p-value)
- **Macierz pomyłek** — wizualna macierz z dokładnością, precyzją, czułością i miarą F1
- **Asystent AI (czat)** — interfejs w języku naturalnym oparty na OpenRouter (Google Gemini) z wywoływaniem narzędzi:
  - *„Dodaj regułę petal.length <= 2.45 Setosa"*
  - *„Usuń regułę 3"*
  - *„Przelicz"*
  - *„Pokaż reguły"*
- **Automatyczne przeliczanie** — dodanie lub usunięcie reguły automatycznie przelicza wszystkie metryki i macierz pomyłek

---

## Stos technologiczny

| Warstwa    | Technologia                                                 |
| ---------- | ----------------------------------------------------------- |
| Frontend   | Angular 19, Tailwind CSS, @ngrx/signals                    |
| Backend    | Python 3.11, FastAPI, scikit-learn, scipy, pandas           |
| Czat AI    | OpenRouter API (Google Gemini 2.0 Flash) z tool calling     |
| Baza danych| SQLite (plikowa, utrwalana przez Docker volume)             |
| Wdrożenie  | Docker Compose (kontenery backend + frontend)               |

---

## Struktura projektu

```
merito/
├── docker-compose.yml          # Konfiguracja Docker Compose
├── .env                        # Klucze API (OPENROUTER_API_KEY, OPENROUTER_MODEL)
├── data/
│   └── irys.csv                # Przykładowy zbiór danych Iris
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # Aplikacja FastAPI (endpointy API, czat AI, wykonywanie narzędzi)
│   └── database.py             # Warstwa bazy danych SQLite (CRUD dla projektów, reguł, macierzy)
├── frontend/
│   ├── Dockerfile
│   ├── angular.json
│   ├── package.json
│   ├── tailwind.config.js
│   └── src/
│       └── app/
│           ├── app.component.ts
│           ├── app.routes.ts
│           ├── services/
│           │   └── api.service.ts          # Klient HTTP do komunikacji z backendem
│           ├── store/
│           │   ├── project.store.ts        # Zarządzanie stanem projektu (reguły, macierz)
│           │   ├── projects.store.ts       # Stan listy projektów
│           │   └── chat.store.ts           # Zarządzanie stanem czatu AI
│           └── components/
│               ├── home/                   # Lista projektów i wgrywanie CSV
│               ├── project/                # Główny widok projektu (reguły + czat)
│               ├── rules-table/            # Tabela reguł z dodawaniem/usuwaniem/przeliczaniem
│               ├── confusion-matrix/       # Wizualizacja macierzy pomyłek
│               └── chat/                   # Panel czatu AI
├── db_data/                    # Wolumen bazy SQLite (utrwalany)
└── uploads/                    # Wgrane pliki CSV
```

---

## Uruchomienie

### Wymagania

- [Docker](https://www.docker.com/) oraz Docker Compose
- Klucz API [OpenRouter](https://openrouter.ai/) (dostępny darmowy plan)

### 1. Sklonuj repozytorium

```bash
git clone <adres-repozytorium>
cd merito
```

### 2. Skonfiguruj zmienne środowiskowe

Utwórz plik `.env` w katalogu głównym projektu:

```env
OPENROUTER_API_KEY=sk-or-v1-twoj-klucz
OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

### 3. Uruchom aplikację

```bash
docker compose up --build
```

| Usługa         | URL                          |
| -------------- | ---------------------------- |
| Frontend       | http://localhost:4200         |
| Backend        | http://localhost:8000         |
| Dokumentacja API | http://localhost:8000/docs  |

### 4. Korzystanie z aplikacji

1. Otwórz http://localhost:4200
2. Wgraj plik CSV (np. dołączony `data/irys.csv`)
3. Wybierz kolumnę docelową (np. `variety`) i kliknij **Train Model**
4. Przejrzyj wygenerowane reguły i macierz pomyłek
5. Użyj **czatu AI**, aby zarządzać regułami w języku naturalnym
6. Użyj przycisków **Add Rule** / **Delete** lub kliknij **Recalculate**, aby zaktualizować metryki

---

## Endpointy API

| Metoda | Endpoint                                    | Opis                                    |
| ------ | ------------------------------------------- | --------------------------------------- |
| GET    | `/api/projects`                             | Lista wszystkich projektów              |
| POST   | `/api/upload`                               | Wgranie CSV i utworzenie projektu        |
| GET    | `/api/projects/{id}`                        | Szczegóły projektu                      |
| POST   | `/api/projects/{id}/classify`               | Trenowanie modelu drzewa decyzyjnego    |
| GET    | `/api/projects/{id}/rules`                  | Pobranie reguł klasyfikacji             |
| POST   | `/api/projects/{id}/rules`                  | Dodanie nowej reguły                    |
| DELETE | `/api/projects/{id}/rules/{rule_id}`        | Usunięcie reguły                        |
| POST   | `/api/projects/{id}/recalculate`            | Przeliczenie metryk i macierzy          |
| GET    | `/api/projects/{id}/confusion-matrix`       | Pobranie macierzy pomyłek               |
| POST   | `/api/projects/{id}/chat`                   | Czat AI z wywoływaniem narzędzi         |

---

## Jak to działa

### Pipeline klasyfikacji

1. Użytkownik wgrywa plik CSV → dane zapisywane w SQLite oraz na dysku
2. Użytkownik wybiera kolumnę docelową → trenowanie drzewa decyzyjnego (scikit-learn)
3. Drzewo jest konwertowane na czytelne reguły IF…THEN
4. Dla każdej reguły obliczane są metryki: **pokrycie** (coverage), **precyzja** (precision), **jakość** (quality), **p-value** (test dwumianowy via scipy)
5. Generowana jest macierz pomyłek z dokładnością, precyzją, czułością i miarą F1

### System czatu AI

Czat wykorzystuje **OpenRouter** (Google Gemini) z mechanizmem **wywoływania narzędzi** (tool calling):

1. Użytkownik wysyła wiadomość → backend buduje prompt systemowy z kontekstem projektu (kolumny, klasy, aktualne reguły, dokładność)
2. Model AI decyduje, które narzędzia wywołać: `get_rules`, `add_rule`, `delete_rule`, `recalculate`, `get_confusion_matrix`
3. Backend wykonuje wywołania narzędzi na bazie danych i zwraca wyniki do AI
4. AI formułuje odpowiedź w języku naturalnym
5. Frontend odświeża tabelę reguł i macierz pomyłek na podstawie wykonanych akcji

---

## Przykładowy zbiór danych

Projekt zawiera plik `data/irys.csv` — klasyczny [zbiór danych Iris](https://pl.wikipedia.org/wiki/Irys_(zbi%C3%B3r_danych)) ze 150 próbkami, 4 cechami (`sepal.length`, `sepal.width`, `petal.length`, `petal.width`) i 3 klasami (`Setosa`, `Versicolor`, `Virginica`).
