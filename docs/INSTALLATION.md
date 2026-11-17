# Installation and Runbook (Windows / VS Code)

## Dataset

The original GroupLens archive is already installed in `data/ml-1m/` in the current workspace. It contains `movies.dat`, `ratings.dat`, and `users.dat`. To install it on another checkout, download `ml-1m.zip` from the [official GroupLens page](https://grouplens.org/datasets/movielens/1m/), place it in `data/`, and run from the project root:

```powershell
Expand-Archive -LiteralPath .\data\ml-1m.zip -DestinationPath .\data
Get-ChildItem .\data\ml-1m\movies.dat, .\data\ml-1m\ratings.dat, .\data\ml-1m\users.dat
```

The downloaded current archive was checked against GroupLens' linked MD5 checksum and ZIP CRC before extraction.

## Backend

```powershell
cd D:\advml\backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API documentation: `http://127.0.0.1:8000/docs`.

## Frontend

Open a second VS Code terminal:

```powershell
cd D:\advml\frontend
npm.cmd install
npm.cmd run dev -- --host 127.0.0.1
```

Website: `http://127.0.0.1:5173`.

## Tests and experiments

From the repository root:

```powershell
python -m unittest discover -s tests -v
python experiments\epsilon_experiments.py
python experiments\evaluate.py
```

The two experiment scripts write actual results to `experiments/results/`. They require all three dataset files. Use distinct terminals for the two servers and stop them with Ctrl+C.
