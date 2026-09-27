from __future__ import annotations

from cleo.application import Application

from livewell import __version__
from livewell.panel.commands import (
    AblateCommand,
    CorrelateCommand,
    CycleCommand,
    ExportCommand,
    SteerCommand,
)


def build_application() -> Application:
    app = Application("livewell", __version__)
    app.add(CycleCommand())
    app.add(SteerCommand())
    app.add(CorrelateCommand())
    app.add(AblateCommand())
    app.add(ExportCommand())
    return app
