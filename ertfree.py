import streamlit as st

def main():
    preprocPage = st.Page("./efreepages/efree_preprocessing.py",
                          title='Preprocessing',
                          icon=":material/app_registration:",
                          url_path='preprocessing')
    procPage = st.Page("./efreepages/efree_processing.py",
                       title='Processing',
                       icon=":material/cycle:",
                       url_path='processing', default=True)
    postProcPage = st.Page("./efreepages/efree_postprocessing.py",
                           title='Postprocessing',
                           icon=":material/chart_data:",
                           url_path='postprocessing')
    pg = st.navigation([preprocPage, procPage, postProcPage],
                       position='top')
    pg.run()

if __name__ == "__main__":
    main()
