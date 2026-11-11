# Scene Lens — Standalone 50-Class Demo

A Flask webpage that classifies uploaded images using ResNet50V2.
It displays the predicted scene, prediction score, and five highest-ranked categories.

Author: Siva Shankar Chodisetti

## Setup and run

Requires Python 3.11. From the repository root, run:

```cmd
cd standalone-demo
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 and upload an image.
Press Ctrl+C to stop the server.

## Model

The included checkpoint must remain beside app.py.
Its filename contains “10classes,” but the checkpoint produces 50 outputs.