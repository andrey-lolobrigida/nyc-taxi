"""Pacote da API FastAPI (inferência + feedback + monitoramento).

Bootstrap de import: adiciona src/ ao sys.path para que a API importe o
MESMO módulo features.py usado no treino (regra de ouro contra
training-serving skew — decisão #3 do CLAUDE.md). Fica no __init__ do
pacote porque qualquer `import api.*` passa por aqui primeiro.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)