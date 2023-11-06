import datetime as dt

from utils import streamlit_frontend_view
# Command Line Interface
class CLI:

    def __init__(self):
        self.execution_timestamp = dt.datetime.utcnow()

    def frontend(self):
        streamlit_frontend_view(
            app_name="App"
        )
