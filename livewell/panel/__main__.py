from __future__ import annotations

from livewell.panel.application import build_application


def main() -> int:
    return int(build_application().run())


if __name__ == "__main__":
    raise SystemExit(main())
