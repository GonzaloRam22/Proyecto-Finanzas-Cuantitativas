import os
import json
import tempfile
import textwrap
import datetime as dt
import platform
from typing import Dict, Optional

from settings import (
    STREAMLIT_DEFAULT_FRONTEND_MODULE_NAME,
    STREAMLIT_DEFAULT_FRONTEND_VIEW_FUNCTION,
)

__all__ = [
    "system_describe",
    "streamlit_frontend_view",
    "google_json_key",
]


def google_json_key(filepath: Optional[str] = None) -> Dict:
    if not filepath:
        filepath = os.path.join(
            os.path.dirname(__file__),
            "google_creds.json"
        )
    # Consideration: reading credentials from the python package is not recommended; use a secret-manager instead.
    with open(filepath, "r") as file:
        return json.loads(file.read())


def streamlit_frontend_view(
        app_name: str,
        frontend_module_name: Optional[str] = None,
        view_function_name: Optional[str] = None,
):
    import importlib
    import sys

    # Replace optional arguments with value
    frontend_module_name = frontend_module_name or STREAMLIT_DEFAULT_FRONTEND_MODULE_NAME
    view_function_name = view_function_name or STREAMLIT_DEFAULT_FRONTEND_VIEW_FUNCTION

    # Create a temporal file
    with tempfile.NamedTemporaryFile(mode="r+", delete=False, suffix=".py") as file:
        file.write(
            textwrap.dedent(
                f"""
                from {app_name}.{frontend_module_name} import {view_function_name}

                if __name__ == "__main__":
                    {view_function_name}()

                """
            )
        )

    streamlit_cli = importlib.import_module("streamlit.web.cli")
    sys.argv = ["streamlit", "run", file.name]
    streamlit_cli.main()


def system_describe(
        enable_cpu_freq: bool = False
) -> Dict:
    import psutil

    uname = platform.uname()
    virtual_memory = psutil.virtual_memory()
    return {
        "timestamp": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "platform_system": uname.system,
        "platform_processor": uname.processor,
        "platform_version": uname.version,
        "virtual_memory_total": virtual_memory.total,
        "virtual_memory_available": virtual_memory.available,
        "virtual_memory_active": virtual_memory.active,
        "virtual_memory_inactive": virtual_memory.inactive,
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        **(
            {} if not enable_cpu_freq else [{
                "cpu_freq_max": cpu_freq.max,
                "cpu_freq_min": cpu_freq.min,
                "cpu_freq_current": cpu_freq.current,
            } for cpu_freq in [psutil.cpu_freq()]][-1]
        )
    }