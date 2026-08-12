import geopandas as gpd
from io import BytesIO
import matplotlib.pyplot as plt
import pyproj
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest
from shapely.geometry import LineString, Point
import streamlit as st

st.set_page_config(layout='wide')

CRS_LIST = pyproj.database.query_crs_info()
CRS_STR_LIST = [f"{crs.auth_name}:{crs.code} - {crs.name}" for crs in CRS_LIST]
CRS_DICT = {f"{crs.auth_name}:{crs.code} - {crs.name}": crs for crs in CRS_LIST}
IL_LIDAR_URL = r"https://data.isgs.illinois.edu/arcgis/services/Elevation/IL_Statewide_Lidar_DEM_WGS/ImageServer/WMSServer?request=GetCapabilities&service=WMS"
GMRT_BASE_URL = r"https://www.gmrt.org:443/services/GridServer?minlongitude&maxlongitude%2C%20&minlatitude&maxlatitude&format=geotiff&resolution=default&layer=topo"
RASTER_SRC_DICT = {"ISGS Statewide Lidar": IL_LIDAR_URL,
                   "Global Multi-Resolution Topography (~30m)": GMRT_BASE_URL,
                   "Other Web Service": "get_service_info",
                   "Raster file": "get_file_name"
                   }

# Establish values for default CRS
DEFAULT_POINTS_CRS = "EPSG:4326 - WGS 84"  # "EPSG:6345 - NAD83(2011) / UTM zone 16N"
DEFAULT_POINTS_CRS_INDEX = CRS_STR_LIST.index(DEFAULT_POINTS_CRS)
DEFAULT_OUTPUT_CRS = DEFAULT_POINTS_CRS


stss = st.session_state
def main():
    st.markdown("PREPROCESSING PAGE IS UNDER CONSTRUCTION")
    me = st.expander("Map Expander", key='map_expander')
    st.session_state.map_container = me.container(width='stretch',
                                                  height=400)
    with st.sidebar:
        st.title('ERFree Preprocessing')
        with st.expander("Data Input", type='compact', expanded=False):
            st.file_uploader("Upload Data file",
                        key='data_uploader')
        st.pills("Topography type",
                options=['None', 'Profile location', 'Topo file'],
                default='None',
                key='topo_type')

        dotopogeo = (st.session_state.topo_type != "None")
        if dotopogeo:
            topoFromLoc = 'location' in stss.topo_type
            with st.expander("Topography and Geolocation", type='compact', expanded=dotopogeo):
                if topoFromLoc:
                    st.file_uploader("Upload Profile Location",
                                    key='profiler_uploader',
                                    on_change=on_upload_profile_loc)
                    crsCol, tTypeCol = st.columns([0.8, 0.2])
                    crsCol.selectbox("Profile CRS",
                            options=CRS_STR_LIST,
                            index=DEFAULT_POINTS_CRS_INDEX,
                            key='profile_crs')
                    tTypeCol.selectbox("Type",
                            options=["Pts", "Line", "Table"],
                            key='topofile_type')
                    
                    
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


def get_elevation(elev_source):

    """This function takes coordinates, specified raster (services), and specified output CRS
       and extracts the elevation values from the raster at the specified coordinates.
       As a streamlit app, this also plots the raster data around the point(s) and the points,
       then displays the points and relevant information in a dataframe that can be copy/pasted.
       
       This was adapted from a standalone python script, so there are inputs/parameters that may
       not be relevant in the streamlit app.
    """
    print("GETTING ELEVATION!!!!!")
    coords=None
    elevation_col_name='elevation'
    xcoord_col_name='xcoord'
    ycoord_col_name='ycoord'
    points_crs=None
    output_crs=None
    elevation_source=None
    elev_source_type='service'
    raster_crs=None
    show_plot=True
    
    #coords = st.session_state.coords
    elevation_col_name = "ELEVATION"
    xcoord_col_name = st.session_state.xcoord_col
    if st.session_state.xcoord_col is None:
        xcoord_col_name = "LONGITUDE"

    ycoord_col_name = st.session_state.ycoord_col
    if ycoord_col_name is None:
        ycoord_col_name = "LATITUDE"

    points_crs = CRS_DICT[st.session_state.well_input_crs].code
    output_crs = 4326
    elev_source_type = 'service'
    raster_crs = None
    print("MADE IT TO THIS PART OF ELEVATION GETTING?")
    #if coords is None:
    #    coordType = st.session_state.coordinate_type
    #    if coordType == "Single":
    #        coords = (st.session_state.xcoord, st.session_state.ycoord)
    #    elif coordType == 'Multiple' or coordType == 'Upload':
    #        coords = st.session_state.point_table
        #if st.session_state.points_source=='Enter coords.':
            #coords = (-88.857362, 42.25637743)

    # Get the correct/specified raster source
    print("CHECKING ESOURCE", hasattr(st.session_state, "elev_source"))
    if "Illinois" in st.session_state.elev_source:
        elevation_source = IL_LIDAR_URL
    else:
        elevation_source = GMRT_BASE_URL

    print("ESOURCE", elevation_source)
    # Get CRS of points, raster, and specified output
    if points_crs is None:
        points_crs = int(CRS_DICT[st.session_state.point_crs].code)
        points_crs_name = CRS_DICT[st.session_state.point_crs].name

    if raster_crs is None:
        raster_crs = int(CRS_DICT[st.session_state.raster_crs].code)
        raster_crs_name = CRS_DICT[st.session_state.raster_crs].name

    if output_crs is None:
        output_crs = int(CRS_DICT[st.session_state.output_crs].code)
        output_crs_name = CRS_DICT[st.session_state.output_crs].name
    elif type(output_crs) is str and output_crs.isnumeric():
        output_crs = int(output_crs)

    # Project coordinates into CRS of raster and specified output CRS
    ptCoordTransformerOUT = pyproj.Transformer.from_crs(crs_from=points_crs,
                                                        crs_to=output_crs,
                                                        always_xy=True)
    ptCoordTransformerRaster = pyproj.Transformer.from_crs(crs_from=points_crs,
                                                           crs_to=raster_crs,
                                                           always_xy=True)

    # Access (geo)dataframe to get location info
    print("DO WE HAVE BUFFER POITNS????", hasattr(st.session_state, 'buffer_points'))
    if hasattr(st.session_state, 'buffer_points'):
        print(type(st.session_state.buffer_points))

    if hasattr(st.session_state, 'buffer_points') and st.session_state.buffer_points is not None:
        # if hasattr(st.session_state, 'well_df_IN') and st.session_state.well_df_IN is not None:
        print("\t yes to buffer points")
        df = st.session_state.buffer_points
        
        coords = df
        xcoord = coords[xcoord_col_name]
        ycoord = coords[ycoord_col_name]
        
        xcoord_OUT, ycoord_OUT = ptCoordTransformerOUT.transform(xcoord, ycoord)
        xcoord_RAST, ycoord_RAST = ptCoordTransformerRaster.transform(xcoord, ycoord)

        minXRast = min(xcoord_RAST)
        maxXRast = max(xcoord_RAST)
        minYRast = min(ycoord_RAST)
        maxYRast = max(ycoord_RAST)

        cols = [f"{points_crs}_xIN", f"{points_crs}_yIN", f"{output_crs}_x", f"{output_crs}_y"]
        dfList = []
        for i, xcoordi in enumerate(xcoord):
            dfList.append([xcoordi, ycoord.iloc[i], xcoord_OUT[i], ycoord_OUT[i]])
        coords = pd.DataFrame(dfList, columns=cols)

        # Get padding for visualization purposes
        xPad = (maxXRast-minXRast)*0.1
        yPad = (maxYRast-minYRast)*0.1

        if float(xPad) == 0.0:
            xPad = maxXRast * 0.01
            xPad = abs(xPad)
            if abs(xPad) > 7500:
                xPad = 7500

        if float(yPad) == 0.0:
            yPad = maxYRast * 0.01
            yPad = abs(yPad)
            if abs(yPad) > 7500:
                yPad = 7500

        rasterXMin = minXRast-xPad
        rasterXMax = maxXRast+xPad

        rasterYMin = minYRast-yPad
        rasterYMax = maxYRast+yPad

        # Read in data from service or file, as appropriate
        # ISGS lidar is from the statewide WGS84 service, read in as an (rio)xarray DataSet
        if "Illinois" in st.session_state.elev_source:
            print("READING FROM ILLINOIS DATA WMS")
            wms = WebMapService(elevation_source)

            bbox = (rasterXMin, rasterYMin, rasterXMax, rasterYMax)

            img = wms.getmap(
                layers=['IL_Statewide_Lidar_DEM_WGS:None'],
                srs='EPSG:3857',
                bbox=bbox,
                size=(256, 256),
                format='image/tiff',
                transparent=True
                )

            bio = BytesIO(img.read())
            elevData_rxr = rxr.open_rasterio(bio)[0]
            elevData_ft = elevData_rxr.rio.reproject(output_crs)
            elevData_m = elevData_ft * 0.3048
            st.session_state.elevation_data = elevData_m
        else:
            # GMRT_URL = r"https://www.gmrt.org:443/services/GridServer?minlongitude=-88.4&maxlongitude=-88.2%2C%20&minlatitude=40.1&maxlatitude=40.3&format=geotiff&resolution=default&layer=topo"
            GMRT_URL = GMRT_BASE_URL.replace('minlongitude', f"minlongitude={rasterXMin:0.4f}")
            GMRT_URL = GMRT_URL.replace('maxlongitude', f"maxlongitude={rasterXMax:0.4f}")
            GMRT_URL = GMRT_URL.replace('minlatitude', f"minlatitude={rasterYMin:0.4f}")
            GMRT_URL = GMRT_URL.replace('maxlatitude', f"maxlatitude={rasterYMax:0.4f}")

            response = requests.get(url=GMRT_URL)
            with BytesIO(response.content) as f:
                elevData_rxr = rxr.open_rasterio(f)
            elevData_m = elevData_rxr.rio.reproject(output_crs)

            if 'band' in elevData_m.dims:
                elevData_m = elevData_m.isel(band=0)

            elevData_ft = elevData_m / 0.3048
            st.session_state.elevation_data = elevData_m

        print("ELEVDATA GOT IT")
        elevData_m[0].plot()
        return elevData_m


def get_utm_crs(point_geometry):
    # Example: coordinates for Chicago, IL
    lon = point_geometry.x
    lat = point_geometry.y
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
            ),
        )

    # The first result is the best match
    utm_crs = utm_crs_list[0]
    st.session_state.utm_crs_code = utm_crs.code
    return utm_crs.code

def ingest_profile():
    profileBytes = st.session_state.profile_uploader.getvalue()
    fileGDF = gpd.read_file(BytesIO(profileBytes))
    st.session_state.profile_from_file = fileGDF.iloc[0].geometry
    st.session_state.profile_gdf = fileGDF
    gdfcrs = fileGDF.crs
    st.session_state.profile_crs = f"{gdfcrs.auth_name}:{gdfcrs.code} - {gdfcrs.name}"


def make_profiles():
    buffer = st.session_state.buffer_size
    if st.session_state.profile_type == 'Upload':
        try:
            profile = st.session_state.profile_from_file
        except Exception:
            st.info("No Profile data has been uploaded. Either use 'Selection' as the Profile input type or upload a geospatial file with LineStrings")
    else:
        profile = st.session_state.current_profile
        if profile is None:
            return
        profile = LineString([(y, x) for x, y in profile.coords])

    profileDict = {"Labels": ["Profile"], 
                   'geometry': [profile]}
    profileGDF = gpd.GeoDataFrame(profileDict,
                                  crs=st.session_state.profile_crs.split(' ')[0])

    profileWGS84 = profileGDF.to_crs("EPSG:4326")
    centroid = profileWGS84.centroid
    utmCRS = get_utm_crs(centroid[0])
    profileUTM = profileGDF.to_crs(utmCRS)


if __name__ == "__main__":
    main()
