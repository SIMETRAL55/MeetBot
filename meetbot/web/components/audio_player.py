"""
Audio player component with seek-to-timestamp capability.

Renders an HTML5 audio player with controls and provides
a method to seek to a specific timestamp (when user clicks a segment).
"""

from pathlib import Path

from nicegui import ui, app


class AudioPlayer:
    """
    Audio player component with seek functionality.

    Wraps HTML5 <audio> element with playback controls
    and programmatic seek-to-timestamp support.
    """

    def __init__(self, audio_url: str) -> None:
        """
        Create an audio player.

        Args:
            audio_url: URL to the audio file (relative or absolute).
        """
        self.audio_url = audio_url
        self._audio_id = f"audio_{id(self)}"

        with ui.column().classes("w-full gap-1"):
            ui.label("Audio Player").classes("text-sm font-medium text-gray-600")

            # HTML audio element
            ui.html(f"""
                <audio id="{self._audio_id}" controls class="w-full"
                       style="width: 100%; border-radius: 8px;">
                    <source src="{audio_url}" type="audio/mpeg">
                    <source src="{audio_url}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            """).classes("w-full")

            # Current time display
            self._time_label = ui.label("0:00 / 0:00").classes(
                "text-xs text-gray-500"
            )

    def seek_to(self, seconds: float) -> None:
        """
        Seek the audio player to a specific timestamp.

        Args:
            seconds: Time in seconds to seek to.
        """
        ui.run_javascript(f"""
            const audio = document.getElementById('{self._audio_id}');
            if (audio) {{
                audio.currentTime = {seconds};
                audio.play();
            }}
        """)

    def pause(self) -> None:
        """Pause audio playback."""
        ui.run_javascript(f"""
            const audio = document.getElementById('{self._audio_id}');
            if (audio) audio.pause();
        """)
