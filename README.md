# MeetBot - Audio Transcription & Speaker Diarization

Turn your audio recordings into searchable transcripts with automated speaker identification. MeetBot combines OpenAI's Whisper for accurate transcription and Pyannote for speaker diarization, then lets you ask questions about the content using semantic search powered by ChromaDB.

## What You Can Do

- **Transcribe audio to text** - Works with MP3, M4A, WAV, and other formats. Runs on your GPU for speed or uses cloud APIs
- **Identify speakers** - Automatically figure out who's speaking and when with speaker diarization
- **Search transcripts** - Index your transcripts and ask natural language questions to find exactly what you need
- **Run everything locally** - Option to keep all inference on your machine with optional local LLMs

## Project Layout

```
meetbot/                        # Main package
├── adapters/                   # Backend implementations
│   ├── transcribers/          # Whisper backends (local, HuggingFace)
│   ├── diarization.py         # Pyannote speaker identification
│   ├── embeddings.py          # Vector encoders for search
│   └── llm/                   # LLM backends (local GGUF, HF API)
├── services/                   # Business logic orchestration
│   ├── transcriber.py         # Transcription with caching & chunking
│   ├── diarizer.py            # Speaker diarization service
│   ├── aligner.py             # Merge transcription with speaker info
│   ├── indexer.py             # Build vector search indexes
│   ├── query_service.py       # RAG-based Q&A
│   └── formatters.py          # Output formatting
├── utils/                      # Helper utilities
│   ├── audio.py               # Format conversion (M4A → WAV)
│   ├── audio_chunker.py       # Break large files into chunks
│   ├── cache.py               # Smart result caching
│   └── chunker.py             # Overlap handling for segments
├── cli.py                      # Command-line interface
├── app.py                      # Pipeline orchestration
└── config.py                   # Configuration management

data/                           # Audio files for processing
results/                        # Output transcripts
db/                            # Vector databases for search
```

## Getting Started

### What You Need

- Python 3.8 or newer
- ffmpeg (for audio format conversion)
- Optional: NVIDIA GPU + CUDA for speed

### Installation (5 minutes)

```bash
# 1. Grab the code
git clone <repository-url>
cd MeetBot

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate           # On Windows: venv\Scripts\activate

# 3. Install everything
pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional: Add your HuggingFace token for API access
echo "HF_API_TOKEN=your_token_here" > .env

# 5. Install the package
pip install -e .

# 6. Verify it's working
python -c "from meetbot.config import settings; print('✓ Ready to go!')"
```

For detailed platform-specific setup, see [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md).

## Using MeetBot

### Transcribe Audio

```bash
# Quick start - transcribe using your GPU
python -m meetbot.cli transcribe data/sample.mp3 \
  --backend local \
  --language en \
  --out results/output.json

# Or use the HuggingFace API
python -m meetbot.cli transcribe data/sample.mp3 \
  --backend huggingface \
  --out results/output.json
```

### Identify Speakers

It happens automatically during transcription. Add speaker constraints:

```bash
python -m meetbot.cli transcribe data/sample.mp3 \
  --backend local \
  --min-speakers 2 \
  --max-speakers 5 \
  --out results/output.json
```

### Search Your Transcript

```bash
# Build a searchable index
python -m meetbot.cli index results/output.json

# Ask questions about it
python -m meetbot.cli query db/output "What did they talk about?"

# Use your local LLM (if set up)
python -m meetbot.cli query db/output "What did they talk about?" --use-local-llm
```

### Available Options

| Option | What It Does | Default |
|--------|-------------|---------|
| `--backend` | Which transcription engine: `local` (GPU) or `huggingface` (API) | huggingface |
| `--language` | Language hint for better accuracy (e.g., 'en', 'es', 'fr') | auto-detect |
| `--out` | Where to save the transcript JSON | results/audio_name.json |
| `--min-speakers` / `--max-speakers` | Diarization constraints | auto-detect |
| `--use-cache` | Reuse previous results if available (faster) | true |
| `--force-refresh` | Ignore cache and re-process everything | false |

## How It Works

The pipeline is straightforward:

```
Your Audio File
      ↓
[Whisper Transcription] → Text with timestamps
      ↓
[Pyannote Diarization] → Speaker segments
      ↓
[Alignment] → Speech from Speaker 1, 2, etc.
      ↓
[Formatting] → Clean JSON output
      ↓
Your Transcript
```

For long files, we automatically split them into overlapping chunks to handle GPU memory limits. Results are cached so you don't repeat expensive operations.

## Transcription Backends

### Local (Recommended for Privacy)

Runs Whisper directly on your GPU. Everything stays on your machine.

```bash
python -m meetbot.cli transcribe data/sample.mp3 --backend local
```

Works best with: NVIDIA GPU + CUDA

### HuggingFace API

Uses cloud API endpoints. Requires HF token but works anywhere.

```bash
export HF_API_TOKEN="hf_..."
python -m meetbot.cli transcribe data/sample.mp3 --backend huggingface
```

## Semantic Search with RAG

After transcription, you can build a searchable index:

```bash
# Step 1: Index the transcript
python -m meetbot.cli index results/output.json

# Step 2: Ask questions
python -m meetbot.cli query db/output "What were the key points?"

# Results include relevant segments + AI-generated answer
```

## Smart Features

### Audio Chunking
Large files get split into overlapping segments automatically. Chunks snap to silence points so words don't get cut off.

### Result Caching
Expensive operations are cached by default. Run the same file twice and the second time is instant. Use `--force-refresh` to skip cache.

### GPU Acceleration
Auto-detects NVIDIA GPU and uses it for Whisper and Pyannote. Falls back to CPU if needed. Much faster with a GPU.

### Multiple Output Formats
Transcripts save as JSON with:
- Text segments with timestamps
- Speaker labels for each segment
- Detected languages
- Metadata about processing

## Troubleshooting

### "ffmpeg not found"
You need the ffmpeg system tool:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### "Out of VRAM" during processing
Reduce audio chunk size or process smaller files. Adjust in `.env`:
```env
AUDIO_CHUNK_SIZE_BYTES=50000000  # Smaller chunks = less memory
```

### "No HuggingFace token found"
Either set the token or use local backend:
```bash
export HF_API_TOKEN="your_token"
# OR
python -m meetbot.cli transcribe data/sample.mp3 --backend local
```

### Slow on CPU?
Switch to GPU backend if you have an NVIDIA card:
```bash
# Install GPU PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then use local backend
python -m meetbot.cli transcribe data/sample.mp3 --backend local
```

## What's Inside

### Key Technologies

- **Whisper**: OpenAI's speech recognition model - works well across languages
- **Pyannote**: Cutting-edge speaker diarization - identifies who's speaking
- **ChromaDB**: Vector database for semantic search
- **LangChain**: Framework for building Q&A systems
- **llama-cpp-python**: Optional local LLM inference (privacy-focused)

### Performance Notes

- **Transcription**: ~1 min audio takes 10-20 seconds on RTX 3050
- **Diarization**: Same time roughly as transcription
- **Indexing**: Fast, depends on document size
- **Search**: Instant retrieval, answer generation takes 1-5 seconds depending on LLM

## Next Steps

1. Read [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md) for your specific setup
2. Check [USAGE.md](USAGE.md) for more advanced examples
3. Look at [RAG_QUICK_REFERENCE.md](RAG_QUICK_REFERENCE.md) for search features
4. Run: `python -m meetbot.cli --help` for all options

## Contributing

Found a bug or want to improve something? Great! Here's how:

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Test it works: `python -m meetbot.cli transcribe data/sample.mp3`
4. Commit with a clear message: `git commit -m "feat: add your feature"`
5. Push and open a pull request


## Questions?

- Check the [issues](../../issues) page
- Look at example commands in this README
- Read the docstrings in the code
- File a new issue if you're stuck

