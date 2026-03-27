from __future__ import annotations

from streamlit_bootstrap import bootstrap_paths

bootstrap_paths()

from caoyao_resnet.streamlit_views import render_home_page


if __name__ == "__main__":
    render_home_page()
