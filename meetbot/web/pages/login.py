"""
Login page for MeetBot web application.

Simple username/password login with validation and error display.
"""

import logging

from nicegui import ui, app

from ..auth import authenticate_user, login_user

logger = logging.getLogger(__name__)


@ui.page("/login")
def login_page() -> None:
    """Render the login page."""

    # If already logged in, redirect to dashboard
    if app.storage.user.get("user_id"):
        ui.navigate.to("/")
        return

    # Center the login form
    with ui.column().classes("absolute-center items-center gap-4"):
        # Header
        ui.label("🎙️ MeetBot").classes("text-4xl font-bold text-blue-800")
        ui.label("Audio Transcription & Speaker Analysis").classes(
            "text-sm text-gray-500 mb-4"
        )

        # Login card
        with ui.card().classes("w-80 p-6"):
            ui.label("Sign In").classes("text-xl font-semibold mb-4")

            username_input = ui.input(
                label="Username",
                placeholder="Enter your username",
            ).classes("w-full").props("outlined")

            password_input = ui.input(
                label="Password",
                placeholder="Enter your password",
                password=True,
                password_toggle_button=True,
            ).classes("w-full").props("outlined")

            error_label = ui.label("").classes(
                "text-red-600 text-sm hidden"
            )

            async def handle_login() -> None:
                """Process login attempt."""
                username = username_input.value.strip()
                password = password_input.value

                if not username or not password:
                    error_label.text = "Please enter username and password"
                    error_label.classes(remove="hidden")
                    return

                user = authenticate_user(username, password)
                if user is None:
                    error_label.text = "Invalid username or password"
                    error_label.classes(remove="hidden")
                    password_input.value = ""
                    return

                login_user(user)
                ui.navigate.to("/")

            login_btn = ui.button(
                "Login",
                on_click=handle_login,
            ).classes("w-full mt-2").props("color=primary")

            # Allow Enter key to submit
            password_input.on("keydown.enter", handle_login)
            username_input.on("keydown.enter", lambda: password_input.run_method("focus"))

        # Footer
        ui.label("Local & Private — All processing on your server").classes(
            "text-xs text-gray-400 mt-4"
        )
