# Commands Cheat Sheet

```bash
# install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# run web
python -m meetbot.cli serve

# run pipeline via CLI
python -m meetbot.cli run path/to/audio.mp3

# query existing index
python -m meetbot.cli query db/<collection> "What are action items?"

# tests
python -m pytest tests/ -v
```
