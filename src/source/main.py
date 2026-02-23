# Backward-compatible entrypoint. Prefer: meetbot.cli.pipeline_cmd
from meetbot.cli.pipeline_cmd import main


if __name__ == "__main__":
    main()
