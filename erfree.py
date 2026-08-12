import importlib
import pathlib
import streamlit as st

eFreeResDir = importlib.resources.files('erfree_resources')
pageDir = pathlib.Path('erfree_resources')

def main():
    preprocPage = st.Page(pageDir.joinpath("erfree_preprocessing.py"),
                          title='Preprocessing',
                          icon=":material/app_registration:",
                          url_path='preprocessing')
    procPage = st.Page(pageDir.joinpath("erfree_processing.py"),
                       title='Processing',
                       icon=":material/cycle:",
                       url_path='processing', default=True)
    postProcPage = st.Page(pageDir.joinpath("erfree_postprocessing.py"),
                           title='Postprocessing',
                           icon=":material/chart_data:",
                           url_path='postprocessing')
    pg = st.navigation([preprocPage, procPage, postProcPage],
                       position='top')
    pg.run()

if __name__ == "__main__":
    main()
