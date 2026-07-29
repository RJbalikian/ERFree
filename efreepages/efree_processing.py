import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import streamlit as st

def main():
    with st.sidebar:
        defaultDataSource = 'Upload data'
        disableUpload = False
        
        if hasattr(st.session_state, 'topo_data') and st.session_state.topo_data is not None:
            hasTopo = True
            disableTopo = True
        
        if hasattr(st.session_state, 'pre_data') and st.session_state.pre_data is not None:
            defaultDataSource = 'Preprocessed data'
            disableUpload = True
            st.checkbox("Use Preprocessed Data", value=True, key='use_pre_data_check')
            if not st.session_state.use_pre_data_check:
                disableUpload = False

        st.pills("Data source", options=['Upload data', 'Preprocessed data'],
                default=defaultDataSource,
                key='data_source')

        if 'Upload' in st.session_state.data_source:
            st.file_uploader('Upload data file',
                        disabled=disableUpload,
                        on_change=on_data_upload,
                        key='data_uploader')
        else:
            st.markdown("Data Preview (not yet available)")
        minRhoCol, maxRhoCol = st.columns([0.5,0.5], vertical_alignment="top")
        minRhoCol.number_input("Discard data not in $$\\rho_{apparent}$$ range:",
                               key='min_rho',
                               value=0)
        maxRhoCol.number_input('',
                               key='max_rho',
                               value=1000)

def on_invert_data():
    data = ert.load(r"\\isgs-sinkhole\geophysics\2DResistivityProjects\BooneCo\BooneCo2026\Data\RUSSELVILLERD_Topo_shift_edit_pyEdit.dat")
    data['k'] = ert.geometricFactors(data)
    # Also make sure error is set (required by the inversion)
    # Use a simple relative error if Var% is not already mapped to 'err'
    errThresh = 0.01
    errPercent = 0.05
    if not data.haveData('err') or all(data['err']==0) or all(data['err']<errThresh) or np.nanmedian(data['err']) < errThresh:
        data['err'] = 0.05

    if st.session_state.quick_range_edit_check:
        data.remove(data['rhoa'] < st.session_state.min_rho)
        data.remove(data['rhoa'] > st.session_state.max_rho)

    mgr = ert.ERTManager(data)
    mgr.data

def on_data_upload():
    if st.session_state.data_uploader is not None:
        print(dir(st.session_state.data_uploader))
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "data.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.session_state.pre_data = ert.load(st.session_state.data_uploader.getvalue())
        data = st.session_state.ert_data = st.session_state.pre_data
        data['k'] = ert.geometricFactors(data)
        st.session_state.min_rho = np.asarray(data['rhoa']).min()
        st.session_state.max_rho = np.asarray(data['rhoa']).max()

    st.session_state.ert_data = data


if __name__ == "__main__":
    main()
