"""Tool interface: a name the model writes in `Action: name[argument]` and a run() method."""

from __future__ import annotations

from typing import Protocol


class Tool(Protocol):
    name: str

    def run(self, argument: str) -> str:
        """Return the observation text for one call."""
        ...
