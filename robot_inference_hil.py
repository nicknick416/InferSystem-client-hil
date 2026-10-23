#!/usr/bin/env python3
"""Standalone HIL entrypoint; no hardware access without --live."""
from HIL.runtime import main

if __name__ == '__main__':
    main()
