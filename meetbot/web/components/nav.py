"""
Navigation header component for MeetBot web application.

Provides a consistent navigation bar across all authenticated pages.
"""

from nicegui import ui, app

from ..auth import logout_user


def create_header() -> None:
    """
    Create the navigation header bar.

    Shows app title, navigation links, and user info with logout.
    """
    display_name = app.storage.user.get("display_name", "User")

    with ui.header().classes("items-center justify-between bg-blue-800 text-white px-6"):
        # Left side: logo and nav
        with ui.row().classes("items-center gap-4"):
            ui.label("🎙️ MeetBot").classes("text-xl font-bold cursor-pointer").on(
                "click", lambda: ui.navigate.to("/")
            )
            ui.separator().props("vertical dark").classes("h-8")
            ui.button(
                "Dashboard",
                icon="dashboard",
                on_click=lambda: ui.navigate.to("/"),
            ).props("flat text-color=white")
            ui.button(
                "Upload",
                icon="upload_file",
                on_click=lambda: ui.navigate.to("/upload"),
            ).props("flat text-color=white")

        # Right side: user info and logout
        with ui.row().classes("items-center gap-2"):
            ui.icon("person").classes("text-xl")
            ui.label(display_name).classes("text-sm")
            ui.button(
                "Logout",
                icon="logout",
                on_click=_handle_logout,
            ).props("flat text-color=white size=sm")


async def _handle_logout() -> None:
    """Handle logout button click."""
    logout_user()
    ui.navigate.to("/login")
