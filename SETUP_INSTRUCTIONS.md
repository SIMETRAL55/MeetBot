# MeetBot Setup & Troubleshooting

## Issue Resolution

### Problem 1: "ModuleNotFoundError: No module named 'meetbot'"

**Cause**: The package needs to be installed so Python can find the `meetbot` module.

**Solution**: Run from the project root directory:

```bash
# From /home/alkris/meetbot/MeetBot/
pip install -e .
```

Or use the setup script:

```bash
chmod +x setup_dev.sh
./setup_dev.sh
```

After installation, these commands will work:

```bash
# Recommended: Use installed package
python -m meetbot.cli transcribe audio.m4a --language ja --backend local
python -m meetbot.cli index results/audio.json
python -m meetbot.cli query db/audio "Question"

# Or use the convenience command (if installed)
meetbot transcribe audio.m4a --language ja --backend local
meetbot index results/audio.json
meetbot query db/audio "Question"
```

### Problem 2: JSON Parsing Error

**Error**: `Invalid JSON on line 1: Expecting property name enclosed in double quotes`

**Cause**: The `prepare_docs` service was expecting a different JSON format than what `format_result_as_json` produces.

**Solution**: ✅ FIXED in latest code

The updated `prepare_docs.py` (lines 91-159) now handles multiple formats:

1. **JSON Array** format: `[{segment1}, {segment2}, ...]`
2. **JSON Object with segments** format (from format_result_as_json):
   ```json
   {
     "input_file": "audio.m4a",
     "duration_seconds": 120.5,
     "segments": [{...}, {...}]
   }
   ```
3. **JSONL** format: one segment per line
4. **Single JSON object** format: `{segment}`

Now the indexing command will work correctly:

```bash
# First transcribe
python -m meetbot.cli transcribe doctor_3.m4a --language ja --backend local
# Output: results/doctor_3.json

# Then index (now works!)
python -m meetbot.cli index results/doctor_3.json
# Automatically detects the JSON format and extracts segments

# Then query
python -m meetbot.cli query db/doctor_3 "Your Japanese question"
```

## Complete Setup Instructions

### Step 1: Install Package

```bash
cd /home/alkris/meetbot/MeetBot
pip install -e .
```

### Step 2: Verify Installation

```bash
python -m meetbot.cli --help
```

You should see the CLI help menu with all available commands.

### Step 3: Test Complete Pipeline

```bash
# Transcribe (converts M4A to WAV automatically)
python -m meetbot.cli transcribe sample_audio.m4a --language ja --backend local

# Index for search
python -m meetbot.cli index results/sample_audio.json

# Ask questions
python -m meetbot.cli query db/sample_audio "Your question in Japanese"
```

## Troubleshooting Checklist

- [ ] Virtual environment is activated: `echo $VIRTUAL_ENV`
- [ ] MeetBot is installed: `pip show meetbot`
- [ ] Python can find the module: `python -c "import meetbot; print(meetbot.__file__)"`
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Result JSON file exists: `ls -la results/`
- [ ] ffmpeg is installed: `ffmpeg -version`
- [ ] pydub is available: `python -c "from pydub import AudioSegment; print('OK')"`

## Manual PYTHONPATH Workaround (if pip install fails)

If you cannot use `pip install -e .`, you can set PYTHONPATH manually:

```bash
# Set for current session
export PYTHONPATH="/home/alkris/meetbot/MeetBot/src:$PYTHONPATH"

# Then use with python -m
python -m src.meetbot.cli transcribe audio.m4a --language ja --backend local
python -m src.meetbot.cli index results/audio.json
python -m src.meetbot.cli query db/audio "Question"
```

Or add to your `.bashrc` or `.zshrc` for permanent setting:

```bash
echo 'export PYTHONPATH="/home/alkris/meetbot/MeetBot/src:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

## Recommended Usage After Installation

```bash
# Simple, clear commands after `pip install -e .`
python -m meetbot.cli transcribe audio.m4a --language ja --backend local
python -m meetbot.cli index results/audio.json
python -m meetbot.cli query db/audio "質問"

# Or if installed as command-line tool:
meetbot transcribe audio.m4a --language ja --backend local
meetbot index results/audio.json
meetbot query db/audio "質問"

# Python API usage
python -c "
from meetbot.app import MeetBotPipeline
pipeline = MeetBotPipeline()
result = pipeline.run('audio.m4a', language='ja')
print('Transcription complete!')
"
```

---

**Both issues are now resolved!** Run `pip install -e .` from the project root and all commands will work correctly.
