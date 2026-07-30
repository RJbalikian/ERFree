
# Session State Attributes
Attributes in the st.session_state variable

| Attribute Name | Default | Data type             | Description |
|----------------|---------|-----------------------|-------------|
| topo_data      | `None`  | gpd.GeoDataFrame      | X, Z and (if known) CRS geometry of topo data |
| profile_loc    | `None`  | gpd.GeoDataFrame      | Label and LineString geometry of 2D ERT profile |
| topo_type      | key     | st.pills              | Whether topo is from "Profile Location" or "Topo file" |
| profile_loc_uploader |key| st.file_uploader      | For geospatial file with profile location | 
| data_uploader |key| st.file_uploader      | For file with ERT data | 
| data_file_name | `None` | str | Filename of uploaded ert data file, after it is uploaded |
| data_df        | `None` | pd.DataFrame | Dataframe version of the data imported from file |