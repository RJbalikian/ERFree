import json
import os
import tempfile

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, LogNorm
from matplotlib import cm
import numpy as np
import pandas as pd
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import streamlit as st

st.set_page_config(page_title='ERTFree Processing',
                   layout='wide',
                   page_icon=":material/cycle:",
                   )


def main():
    with st.sidebar:
        defaultDataSource = 'Upload data'
        disableUpload = False
        invButtonDisabled = True

        if hasattr(st.session_state, 'ert_data') and st.session_state.ert_data is not None:
            invButtonDisabled = False
            st.text(f"Loaded file: {st.session_state.data_file_name:.50}")

        st.button("Run Inversion", 
                  type='primary', 
                  width='stretch',
                  on_click=on_invert_data,
                  disabled=invButtonDisabled,
                  key='run_inverse_button')

        if hasattr(st.session_state, 'topo_data') and st.session_state.topo_data is not None:
            hasTopo = True
            disableTopo = True
        
        if hasattr(st.session_state, 'pre_data') and st.session_state.pre_data is not None:
            defaultDataSource = 'Preprocessed data'

        st.pills("Data source", options=['Upload data', 'Preprocessed data'],
                default=defaultDataSource,
                key='data_source')

        if 'Upload' in str(st.session_state.data_source) or st.session_state.data_source is None:
            st.file_uploader('Upload data file',
                        disabled=disableUpload,
                        on_change=on_data_upload,
                        key='data_uploader')
        else:
            st.markdown("Data Preview (not yet available)")
            st.session_state.data_preview_container = st.container()
        qeExpanded = False
        hasERTData = False
        if hasattr(st.session_state, 'ert_data') and st.session_state.ert_data is not None:
            qeExpanded = True
            hasERTData = True

        with st.expander("Quick edits", key='quickedit_expander',
                    expanded=qeExpanded):

            # App Rho Range
            st.checkbox("Discard data not in $$\\rho_{apparent}$$ range:",
                        value=False, key='quick_apprho_range')
            disableAppRhoRange = True
            if st.session_state.quick_apprho_range:
                disableAppRhoRange = False
            blankRhoCol, minRhoCol, maxRhoCol = st.columns([0.1, 0.5,0.5], vertical_alignment="top")

            minRhoVal = 0
            maxRhoVal = 1000
            if hasERTData:
                minRhoVal = np.asarray(st.session_state.ert_data['rhoa']).min()
                maxRhoVal = np.asarray(st.session_state.ert_data['rhoa']).max()

            minRhoCol.number_input("Min $$\\rho_{apparent}$$",
                                   key='min_rho',
                                   value=minRhoVal, 
                                   disabled=disableAppRhoRange)

            maxRhoCol.number_input('Max $$\\rho_{apparent}$$',
                                   disabled=disableAppRhoRange,
                                   key='max_rho',
                                   value=maxRhoVal)

            # Data Level % Err Threshold
            dataDF = st.session_state.data_df = calculate_data_level_errors()

            if dataDF is None or 'DL_err' not in dataDF.columns:
                maxPctError = 1000
            else:
                maxPctError = dataDF['DL_Err'].max()
            if maxPctError > 1000:
                maxPctError = 1000

            st.slider(r"Data Level % Error Threshold",
                      min_value=0,
                      max_value=maxPctError,
                      value=(0, maxPctError),
                      on_change=pct_error_update,
                      key='pct_err_slider')


def on_invert_data():
    print('inverting data')
    st.write("Inverting data")
    data = st.session_state.ert_data
    data['k'] = ert.geometricFactors(data)
    print(data['k'])
    # Also make sure error is set (required by the inversion)
    # Use a simple relative error if Var% is not already mapped to 'err'
    errThresh = 0.01
    errPercent = 0.05
    if not data.haveData('err') or any(data['err']==0) or all(data['err']==0) or all(data['err']<errThresh) or np.nanmedian(data['err']) < errThresh:
        data['err'] = errPercent

    if st.session_state.quick_apprho_range:
        data.remove(data['rhoa'] < st.session_state.min_rho)
        data.remove(data['rhoa'] > st.session_state.max_rho)

    mgr = ert.ERTManager(data)
    sensors = np.array(data.sensors())

    a = np.array(data['a'], dtype=int)   # current electrode +
    b = np.array(data['b'], dtype=int)   # current electrode -
    m = np.array(data['m'], dtype=int)   # potential electrode +
    n = np.array(data['n'], dtype=int)   # potential electrode -

    sensorStack = np.stack([sensors[a, 0], sensors[b, 0] ,sensors[m, 0] ,sensors[n, 0]]).T

    mesh = ert.createInversionMesh(
                data,
                paraDX=0.5, # smaller = finer horizontal cells
                paraDepth=100, # shallower model depth
                quality=34
                )

    mgr = ert.ERTManager(data)
    st.session_state.mgr = mgr
    inv = mgr.invert(mesh=mesh, maxIter=5, verbose=True)
    st.session_state.inv = inv

    obs = np.asarray(mgr.inv.dataVals)
    calc = np.asarray(mgr.inv.response)
    percent_error = np.nanmean(np.abs(obs - calc) / obs) * 100
    
    obs = np.asarray(mgr.inv.dataVals)
    calc = np.asarray(mgr.inv.response)
    mape = np.mean(np.abs(obs - calc) / obs * 100)

    print("Chi²", mgr.inv.chi2())
    print("% Error", mape)
    
    show_threeplot_results()


def on_data_upload():
    if st.session_state.data_uploader is not None:
        print(dir(st.session_state.data_uploader))
        st.session_state.data_file_name = st.session_state.data_uploader.name
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.txt")
            with open(data_path, "wb") as f:
                f.write(st.session_state.data_uploader.getbuffer())
            st.session_state.pre_data = ert.load(data_path)
        data = st.session_state.ert_data = st.session_state.pre_data
        data['k'] = ert.geometricFactors(data)
        st.session_state.min_rho = np.asarray(data['rhoa']).min()
        st.session_state.max_rho = np.asarray(data['rhoa']).max()
        #fig, ax = plt.subplots()
        #ax.scatter(data.sensors()[:, 0], data.sensors()[:, 1])
        #st.session_state.data_preview_container.pyplot(fig)

        st.session_state.ert_data = data


def pct_error_update():
    st.session_state.pct_err_slider = (0, st.session_state.pct_err_slider[1])


def calculate_data_level_errors():
    # Get dataframe to manipulate
    if not hasattr(st.session_state, 'ert_data') or st.session_state.ert_data is None:
        return
    elif not hasattr(st.session_state, 'data_df') or st.session_state.data_df is None:
        dataDF = st.session_state.data_df = get_df_from_data(st.session_state.ert_data)
    else:
        dataDF = st.session_state.data_df
        
    if 'DL_Err' not in dataDF.columns:
        # Calculate data levels, and mean values for each level, based on a and n
        uniqueAspace = set(dataDF['aSpace'].unique())
        uniqueNfactor = set(dataDF['nFactor'].unique())

        currDL = 1
        dataDF['DataLevel'] = np.nan
        dataDF['DL_Mean'] = np.nan
        for aspace in uniqueAspace:
            for nfactor in uniqueNfactor:
                dlDF = dataDF[(dataDF['aSpace']==aspace) & (dataDF['nFactor']==nfactor)]
                if dlDF.shape[0] > 0:
                    dataDF.loc[dlDF.index, 'DataLevel'] = currDL
                    rCol = 'rhoa'
                    if 'rhoa' not in dlDF.columns or dlDF['rhoa'].isnull().all():
                        rCol = 'r'
                    dlMean = np.round(dlDF[rCol].mean(), 5)
                    dataDF.loc[dlDF.index, 'DL_Mean'] = dlMean
                    currDL += 1
        dataDF['DL_Err'] = np.round(((dataDF['DL_Mean'] - dataDF[rCol]) / dataDF['DL_Mean']) * 100, 5)

    return dataDF

def get_df_from_data(data):
    sensor_x = data.sensors()[:, 0]
    a_x = sensor_x[data["a"]]
    b_x = sensor_x[data["b"]]
    m_x = sensor_x[data["m"]]
    n_x = sensor_x[data["n"]]
    elecLocs = np.stack([a_x, b_x, m_x, n_x]).T

    df = pd.DataFrame(elecLocs, columns=['Ax', 'Bx', 'Mx', 'Nx'])
    df['pseudoX'] = df[['Ax', 'Nx']].mean(axis=1)
    df['pseudoZ'] = np.round((np.tan(np.deg2rad(28)) * np.abs(df['Ax'] - df['Nx']) / 2), 2)
    df['aSpace'] = np.abs(df['Ax'] - df['Bx'])
    df['nFactor'] = (df[['Ax', 'Bx']].mean(axis=1) - df[['Mx', 'Nx']].mean(axis=1)) // df['aSpace']

    mayNotHave = ['r', 'rhoa', 'stacks', 'err', 'u', "Latitude", "Longitude"]
    for col in mayNotHave:
        if col in data.dataMap().keys():
            df[col] = data[col]
    
    return df

def show_threeplot_results():
    data = st.session_state.ert_data
    mgr = st.session_state.mgr
    inv = st.session_state.inv
    # ── Extract sensor positions ─────────────────────────────────────────────────
    sensors = np.array(data.sensors())  # shape (N, 2) → x, z columns

    # ── Extract electrode indices (0-based) ──────────────────────────────────────
    a = np.array(data['a'], dtype=int)   # current electrode +
    b = np.array(data['b'], dtype=int)   # current electrode -
    m = np.array(data['m'], dtype=int)   # potential electrode +
    n = np.array(data['n'], dtype=int)   # potential electrode -

    sensorStack=np.stack([sensors[a, 0], sensors[b, 0] ,sensors[m, 0] ,sensors[n, 0]])

    # ── Pseudosection coordinates (midpoint / pseudo-depth convention) ────────────
    mid_x  = (sensors[a, 0] + sensors[b, 0] + sensors[m, 0] + sensors[n, 0]) * 0.25
    pseudo_z = (np.nanmax(sensorStack, axis=0) - np.nanmin(sensorStack, axis=0)) * -0.25


    # ── Resistivity values ────────────────────────────────────────────────────────
    rhoa_obs = np.array(data['rhoa'])                  # observed apparent resistivity
    rhoa_fwd = np.array(mgr.inv.response)              # forward modelled apparent resistivity

    # ── Inverted model on mesh ────────────────────────────────────────────────────
    # Node coordinates
    pmesh = mgr.paraDomain
    model_xCenter = np.array([c.center().x() for c in pmesh.cells()])
    model_zCenter = np.array([c.center().y() for c in pmesh.cells()])
    rho_inv = np.asarray(inv)

    mesh_x = np.array(mgr.mesh.cellCenters())[:, 0]
    mesh_z = np.array(mgr.mesh.cellCenters())[:, 1]


    # ── Triangulation for pseudosection panels ───────────────────────────────────
    triang_pseudo = tri.Triangulation(mid_x, pseudo_z)

    # ── Triangulation for model panel ────────────────────────────────────────────
    triang_model = tri.Triangulation(mesh_x, mesh_z)

    # ── Plot ──────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    vmin = min(rhoa_obs.min(), rhoa_fwd.min())
    vmax = max(rhoa_obs.max(), rhoa_fwd.max())

    # Panel A — observed
    ax = axes[0]
    tc0 = ax.tripcolor(triang_pseudo, rhoa_obs, shading='flat',
                    cmap='jet', vmin=vmin, vmax=vmax)
    ax.scatter(mid_x, pseudo_z, c=rhoa_obs, cmap='jet',
            vmin=vmin, vmax=vmax, s=20, edgecolors='k', linewidths=0.3, zorder=3)
    plt.colorbar(tc0, ax=ax, label='Apparent Resistivity (Ω·m)')
    ax.set_title('a) Observed Apparent Resistivity')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Pseudo-depth (m)')

    # Panel B — forward model response
    ax = axes[1]
    tc1 = ax.tripcolor(triang_pseudo, rhoa_fwd, shading='flat',
                    cmap='jet', vmin=vmin, vmax=vmax)
    ax.scatter(mid_x, pseudo_z, c=rhoa_fwd, cmap='jet',
            vmin=vmin, vmax=vmax, s=20, edgecolors='k', linewidths=0.3, zorder=3)
    plt.colorbar(tc1, ax=ax, label='Apparent Resistivity (Ω·m)')
    ax.set_title('b) Forward Model Apparent Resistivity')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Pseudo-depth (m)')

    # Panel C — inverted model

    # ── Sensitivity / coverage ───────────────────────────────────────────────────
    # mgr.coverage() -> log10 sensitivity per paraDomain cell (same order as mgr.model)
    cov = mgr.coverage()

    # Normalize coverage to [0, 1] for alpha, clipping extreme tails so a few
    # very high/low sensitivity cells don't wash out the whole scale
    cov_lo, cov_hi = np.percentile(cov, [0, 100])
    alpha_min = 0.01  # fully "invisible" cells still get a faint tint
    alpha = np.clip((cov - cov_lo) / (cov_hi - cov_lo), 0, 1)
    alpha = alpha_min + (1 - alpha_min) * alpha

    # ── Colors from resistivity (log-scaled) ─────────────────────────────────────
    norm = LogNorm(vmin=rho_inv.min(), vmax=rho_inv.max())
    cmap = plt.get_cmap('nipy_spectral')
    rgba = cmap(norm(rho_inv))
    #rgba[:, 3] = alpha   # overwrite alpha channel with sensitivity-based transparency
    nodes = np.array(mgr.paraDomain.positions())

    plot_type = 'model cells'
    plot_type = 'contour'
    ax = axes[2]
    # ── Build cell polygons directly from the mesh (works for tris and quads) ───
    if 'model' in plot_type:
        polys = [nodes[np.array(c.ids())][:, :2] for c in mgr.paraDomain.cells()]

        pc = PolyCollection(polys, facecolors=rgba, edgecolors='none')
        ax.add_collection(pc)
        ax.autoscale_view()
    elif 'cont' in plot_type:
        print(mid_x.shape, pseudo_z.shape, rho_inv.shape)
        ax.tricontourf(model_xCenter, model_zCenter, np.log10(rho_inv),
                    levels=18, cmap=cmap)

    # Colorbar needs its own ScalarMappable since PolyCollection alpha is manual
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Modeled Resistivity (Ω·m)')

    # ── Depth limit informed by sensitivity ──────────────────────────────────────
    cellCenters = np.array(mgr.paraDomain.cellCenters())
    scov = mgr.standardizedCoverage(threshold=-3.5)   # 0/1 per cell
    well_resolved_z = cellCenters[scov > 0, 1]

    max_depth_cap = 100
    if well_resolved_z.size > 0:
        sens_depth_limit = np.percentile(well_resolved_z, 0.02)  # 2nd percentile = deep edge of good coverage
    else:
        sens_depth_limit = max_depth_cap # fallback

    # Don't show deeper than ~100 m even if a few cells claim coverage beyond that,
    # but don't force 100 m if sensitivity runs out sooner (shallow line, thin mesh, etc.)
    topo_min = sensors[:, 1].min()
    depth_limit = min(sens_depth_limit, max_depth_cap)  # less negative of the two = shallower cutoff wins if sensitivity is worse than 100 m
    yPad = depth_limit * 0.05
    minY = topo_min - (depth_limit - yPad)
    maxY = sensors[:, 1].max() + yPad

    ax.plot(sensors[:, 0], sensors[:, 1], c='green', linewidth=2)
    ax.scatter(sensors[:, 0], sensors[:, 1], c='k', marker='v', s=10, edgecolors='None', zorder=100)

    # Sort by x
    pts = np.column_stack((mid_x, pseudo_z))
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    # Find leftmost and rightmost hull vertices
    left = np.argmin(hull_pts[:, 0])
    right = np.argmax(hull_pts[:, 0])

    # Walk both directions around the hull
    if left <= right:
        path1 = hull_pts[left:right+1]
        path2 = np.vstack((hull_pts[right:], hull_pts[:left+1]))
    else:
        path1 = np.vstack((hull_pts[left:], hull_pts[:right+1]))
        path2 = hull_pts[right:left+1]

    # The lower hull path has lower mean z
    lower_hull = path1 if np.mean(path1[:, 1]) < np.mean(path2[:, 1]) else path2

    # Sort by x for interpolation
    lower_hull = lower_hull[np.argsort(lower_hull[:, 0])]

    bottom_fun = interp1d(
            lower_hull[:, 0],
            lower_hull[:, 1],
            bounds_error=False,
            fill_value=np.nan
            )

    sensorTops = sensors[:, 1]
    sensorBottoms = bottom_fun(sensors[:, 0])
    xFill = sensors[:, 0]
    xFill = xFill.tolist()
    xFill.insert(0, min(nodes[:, 0]))
    xFill.append(max(nodes[:, 0]))
    yTop = sensors[:, 1] + sensorBottoms
    yTop[np.isnan(yTop)] = max(nodes[:, 1])
    yTop = yTop.tolist()
    yTop.insert(0, max(nodes[:, 1]))
    yTop.append(max(nodes[:, 1]))
    yBottom = np.zeros_like(yTop)+min(nodes[:, 1])
    ax.fill_between(xFill, 
                    yTop,
                    yBottom, 
                    facecolor='white',
                    zorder=1000)
    minY = min(yTop) - yPad

    ax.set_ylim(minY, maxY)
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())

    ax.set_title('c) Inverted Resistivity Model')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')

    plt.tight_layout(pad=2.5)
    st.selectbox("Download data",
                  options=['JSON', 'Plot (3x)', "Plot (Model)", "VTK"],
                  key="dl_type_select"
                  )

    jsonText = convert_to_json()

    st.download_button(
        label="JSON",
        data=jsonText,
        file_name=f"INV_{st.session_state.data_file_name}.json",
        mime="application/json",
        icon=":material/data_object:"
        )

    #jsonDict = {'test':"value"}
    #kwargDict = {
    #    'JSON':{'label': 'JSON',
    #            'data': jsonDict,
    #            'file_name': fname,
    #            'mime': 'application/json'
    #            }
    #}
    #dlKwargs = kwargDict[st.session_state.dl_type_select]
    #st.download_button(type='tertiary',
    #                   **dlKwargs
    #                   )
    #st.download_button('3x Plot')
    #st.download_button('Model Plot')
    st.pyplot(fig)

# JSON File
def convert_to_json():
    dataDF = get_df_from_data(st.session_state.ert_data)
    attrsToCheck = ['ert_data', 'mgr', 'inv']
    mgr = st.session_state.mgr
    pmesh = mgr.paraDomain
    inv = st.session_state.inv

    rCol = 'rhoa'
    if 'rhoa' not in dataDF or dataDF['rhoa'].isnull().all():
        try:
            dataDF['rhoa'] = np.asarray(mgr.inv.dataVals)
        except Exception:
            rCol = 'r'


    # Get Chi2 data
    ch2Total = mgr.inv.chi2History
    iterations = [int(i) for i in np.arange(len(ch2Total))]
    chi2Dict = dict(zip(iterations, ch2Total))

    # Model iterations
    allModels = mgr.inv.modelHistory
    iterations = [int(i) for i in np.arange(len(allModels))]
    modelDict = dict(zip(iterations, np.asarray(allModels).tolist()))

    jsonDict = {"Profile_Name": st.session_state.data_file_name,
                "Project_Name": None,
                "XYZ": None,
                "Array": None,
                "Spread": None,
                "Min_Elec_Spacing": None,
                'Data_Locations': dataDF[['pseudoX', 'pseudoZ']].to_dict(),
                "Data_Observed": dataDF[rCol].to_numpy().tolist(),
                "Data_Forward": np.asarray(mgr.inv.response).tolist(),
                "Data_Residuals": np.asarray(mgr.inv.residual()).tolist(),
                "Model_Locations": {'MeshXCenter': [c.center().x() for c in pmesh.cells()],
                                    'MeshZCenter': [c.center().y() for c in pmesh.cells()]},
                "Model_Resistivity": modelDict,
                "Chi2": chi2Dict,
                }
    
    return json.dumps(jsonDict)


if __name__ == "__main__":
    main()
