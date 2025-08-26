# Titanic Itinerary Dashboard

A data visualization dashboard for exploring Titanic passenger data, built with Python,
Dash, and Plotly.

## Features

- **Interactive Dashboard:**  
  Visualizes Titanic passenger clusters on a ship diagram, with dynamic filtering by
  scenario.
- **Category Filter:**  
  Select from custom passenger scenarios (e.g., Social Climber, Last Minute Ticket) to
  update all metrics and the ship map.
- **Summary Metrics:**  
  Displays survival percentage, number of males, number of females, and average age for
  the selected group.
- **Responsive Layout:**  
  Clean, modern UI with equispaced info cards and a custom ship background.

## Quickstart

### 0. Download Data

It's a csv, and you can get it from [here](https://www.kaggle.com/competitions/titanic/data)

### 1. Install Dependencies

> [!NOTE]
>
> This assumes that you have pkgx...if not, see [here](https://pkgx.sh)

This project uses [astral.sh/uv](https://astral.sh/uv/) for dependency management

```sh
pkgx uv pip install -r pyproject.toml
```

### 2. Run the App

```sh
pkgx uv run python main.py
```

The app will be available at [http://127.0.0.1:8050](http://127.0.0.1:8050).

### 3. Project Structure

```
assets/
  - layout.css, typography.css, ship.png, etc.
data/
  - titanic.csv, test.csv
main.py
pyproject.toml
pkgx.yaml
README.md
```

- **main.py**: Main Dash app and callbacks
- **assets/**: Static assets (images, CSS)
- **data/**: Titanic dataset (CSV)

## Development

- **Linting:**  
  Uses `ruff` for Python linting.
  ```sh
  pkgx ruff check .
  ```

## Contributing

- Fork and clone the repo.
- Install dependencies with `uv`.
- Make your changes and submit a pull request.

## Notes

- The dashboard is designed for technical users and contributors.
- All dependencies are managed via `uv` and `pkgx`—do not use pip directly.

---

**Questions?**  
Open an issue or contact the maintainer.
