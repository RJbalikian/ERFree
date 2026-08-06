import geopandas as gpd
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit as st
st.set_page_config(layout='wide')

stss = st.session_state
def main():
    st.markdown("PREPROCESSING PAGE IS UNDER CONSTRUCTION")
    me = st.expander("Map Expander", key='map_expander')
    st.session_state.map_container = me.container(width='stretch',
                                                  height=400)

    with st.sidebar:
        st.file_uploader("Upload Data file")
        st.pills("Topography type",
                options=['Profile location', 'Topo file'],
                default='Profile location',
                key='topo_type')

        if 'location' in stss.topo_type:
            st.file_uploader("Upload Profile Location",
                             key='profile_loc_uploader',
                            on_change=on_upload_profile_loc)
            stss.topo_data = None
            stss.profile_loc = None
        else:
            st.file_uploader("Upload Topographic Data")
            stss.profile_loc = None


def on_upload_profile_loc():
    if stss.profile_loc_uploader is not None:
        stss.topo_data = gpd.read_file(stss.profile_loc_uploader.getvalue())
        fig, ax = plt.subplots(figsize=(5,5))
        stss.topo_data.plot(ax=ax)
        st.session_state.map_container.pyplot(fig)

if __name__ == "__main__":
    main()
