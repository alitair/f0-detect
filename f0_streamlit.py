import pandas as pd
import streamlit as st
import os
import glob
import json


from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype
st.set_page_config(layout="wide")

BASE_DIR = "/home/alistairfraser/data/buckets/oregon.birdconv.mp4/tweety/"

@st.cache_data
def load_data():

    df = pd.read_json("http://birdcallauth.web.app/api/rooms/FOjDCLnGtLWYAEeFIpJYuUMWrNJ2/tracks")

    df["filepath"]   = pd.NA
    df["f0"]         = pd.NA
    df["name_index"] = pd.NA

    if os.path.exists(BASE_DIR):
        for room in df["room_name"].unique():
            room_dir = os.path.join(BASE_DIR, room)
            if os.path.exists( room_dir ):
                mp4_files = glob.glob( os.path.join(room_dir, "*.mp4") )  
                if mp4_files:
                    df.loc[df["room_name"] == room, "filepath"] = mp4_files[0] 

                    filename = os.path.basename(mp4_files[0]).replace(".mp4", "")
                    parts = filename.split("-")
                    
                    if len(parts) > 1:  # Ensure there are names after the timestamp
                        extracted_names = parts[1:]  # All elements after the timestamp

                        # Assign `name_index` based on position in the filename
                        for idx, name in enumerate(extracted_names):
                            df.loc[(df["room_name"] == room) & (df["name"] == name), "name_index"] = idx

                f0_data = glob.glob( os.path.join(room_dir, "*_f0.json") ) 
                if f0_data:
                    if os.path.exists(f0_data[0]):
                        with open(f0_data[0], "r") as f:
                            df.loc[df["room_name"] == room, "f0"] = json.load(f)

    return df

# Load and cache the dataset
df = load_data()

if "selected_room" not in st.session_state:
    st.session_state.selected_room = None  # Start with no selection

# Convert `start_ts` from Unix timestamp (seconds) to Pandas datetime
df["start_ts"] = pd.to_datetime(df["start_ts"], unit="s")

# Convert datetime to integer timestamps (seconds) for calculations
df["start_ts_unix"] = df["start_ts"].astype(int) // 10**9  # Convert nanoseconds to seconds

# Ensure duration is numeric and handle NaNs



# Convert `duration` to minutes:seconds format for display
def format_duration(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02d}"

df["length"] = df["duration"].apply(format_duration)

df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int) // 60 # convert to minutes


# Sidebar Filters
st.sidebar.header("Conversation Filters")

# Include filters
name_options = df["name"].dropna().unique()
species_options = df["species"].dropna().unique()

st.sidebar.markdown("### Include calls with:")
name_selected = st.sidebar.multiselect("birds", name_options, placeholder="Select to include")
species_selected = st.sidebar.multiselect("species", species_options, placeholder="Select to include")


# **Apply Include Filtering Logic**
if name_selected or species_selected:
    room_names_to_include = set(df["room_name"].unique())  # Start with all rooms

    if name_selected:
        name_room_sets = [
            set(df.loc[df["name"] == name, "room_name"]) for name in name_selected
        ]
        room_names_to_include &= set.intersection(*name_room_sets) if name_room_sets else set()

    if species_selected:
        species_room_sets = [
            set(df.loc[df["species"] == species, "room_name"]) for species in species_selected
        ]
        room_names_to_include &= set.intersection(*species_room_sets) if species_room_sets else set()

    # Filter dataset to include only matching room_names
    df = df[df["room_name"].isin(room_names_to_include)]

    # Exclude filters
st.sidebar.markdown("### Exclude calls with:")    
name_options = df["name"].dropna().unique()
species_options = df["species"].dropna().unique()
name_excluded = st.sidebar.multiselect("birds", name_options, placeholder="Select to exclude")
species_excluded = st.sidebar.multiselect("species", species_options, placeholder="Select to exclude")


# **Apply Exclude Filtering Logic**
if name_excluded or species_excluded:
    room_names_to_exclude = set()

    if name_excluded:
        name_room_sets_exclude = [
            set(df.loc[df["name"] == name, "room_name"]) for name in name_excluded
        ]
        room_names_to_exclude |= set.union(*name_room_sets_exclude) if name_room_sets_exclude else set()

    if species_excluded:
        species_room_sets_exclude = [
            set(df.loc[df["species"] == species, "room_name"]) for species in species_excluded
        ]
        room_names_to_exclude |= set.union(*species_room_sets_exclude) if species_room_sets_exclude else set()

    # Filter dataset to exclude matching room_names
    df = df[~df["room_name"].isin(room_names_to_exclude)]


if not df.empty:
    st.sidebar.markdown("### Include calls between:")
    duration_min = int(df["duration"].min())
    duration_max = int(df["duration"].max())
    if duration_min < duration_max:
        duration_range = st.sidebar.slider(
            "Duration (minutes)",
            min_value=duration_min,
            max_value=duration_max,
            value=(duration_min, duration_max)
        )
        df = df[df["duration"].between(duration_range[0], duration_range[1])]

if not df.empty:
    # `start_ts` filter (date picker)
    start_date = st.sidebar.date_input(
        "Start Date (From - To)", 
        value=(df["start_ts"].min().date(), df["start_ts"].max().date()),
    )

    df = df[(df["start_ts"].dt.date >= start_date[0]) & (df["start_ts"].dt.date <= start_date[1])]

if not df.empty:
    # `room_name` filter (search for rooms that start with text)
    room_name_search = st.sidebar.text_input("Find rooms that start with")
    if room_name_search:
        df = df[df["room_name"].str.startswith(room_name_search, na=False)]

if not df.empty:
    # `video_bitrate` filter (category)
    video_bitrate_selected = st.sidebar.multiselect(
        "Video Bitrate", df["video_bitrate"].unique(), default=df["video_bitrate"].unique()
    )
    df = df[df["video_bitrate"].isin(video_bitrate_selected)]

if not df.empty:
    # `audio_bitrate` filter (category)
    audio_bitrate_selected = st.sidebar.multiselect(
        "Audio Bitrate", df["audio_bitrate"].unique(), default=df["audio_bitrate"].unique()
    )
    df = df[df["audio_bitrate"].isin(audio_bitrate_selected)]

if not df.empty:
    # `agc_enabled` filter (category dropdown)
    agc_enabled_selected = st.sidebar.multiselect(
        "Automatid Gain Control", df["agc_enabled"].unique(), default=df["agc_enabled"].unique()
    )
    df = df[df["agc_enabled"].isin(agc_enabled_selected)]


if not df.empty:
    # `bypass_voice_processing` filter (category dropdown)
    bypass_voice_processing_selected = st.sidebar.multiselect(
        "Bypass Voice Processing", df["bypass_voice_processing"].unique(), default=df["bypass_voice_processing"].unique()
    )
    df = df[df["bypass_voice_processing"].isin(bypass_voice_processing_selected)]


# Ensure `duration` is numeric to avoid errors
# Ensure `duration` is numeric to avoid errors
if "duration" in df.columns and not df.empty:
    total_seconds = df["duration"].sum() * 60  # Convert minutes to seconds
    total_hours = total_seconds // 3600
    total_minutes = (total_seconds % 3600) // 60
    formatted_total_duration = f"{int(total_hours)} hours {int(total_minutes):02d} minuntes"
    num_unique_rooms = df["room_name"].nunique()
else:
    formatted_total_duration = "<>"  # Default if no data
    num_unique_rooms = 0


# **Hide columns**
colunns_to_drop = ["id", "directory", "start_ts_unix", "video_hevc","duration","max_participants"]
df = df.drop(columns=colunns_to_drop, errors="ignore")


# Define column order
column_order = [
    "name",
    "species",
    "length",  
    "start_ts",
    "room_name",
    "video_bitrate",
    "audio_bitrate",
    "agc_enabled",
    "bypass_voice_processing",
    "filepath",
    "f0"
]

# Reorder columns (only keeping existing ones)
df = df[[col for col in column_order if col in df.columns]]


# Define new column names
column_rename_map = {
    "name": "Bird",
    "species": "Species",
    "duration": "minutes",
    "start_ts": "Start",
    "room_name": "Room",
    "video_bitrate": "Video",
    "audio_bitrate": "Audio",
    "agc_enabled": "AGC",
    "bypass_voice_processing": "BVP"
}
df = df.rename(columns=column_rename_map)


# Configure column settings to hide specific columns
columns_to_hide = ["filepath", "f0", "name_index" ]
column_config = {col: {"hidden": True} for col in columns_to_hide}


def on_change_callback():
    selection = st.session_state.my_editor['selected_rows']
    if selection:
        # Get the index of the selected row
        selected_index = selection[0]
        # Get the selected row data
        selected_row = df.iloc[selected_index]
        st.write("Selected row:", selected_row)


# Allow user to select a row
event = st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_config, 
    on_select="rerun",
    selection_mode="single-row")

# Display total duration and unique rooms
st.markdown(f"{num_unique_rooms} conversations. {formatted_total_duration}")

# Get the selected rows

if event.selection.rows:
    selected_row_index = event.selection.rows[0]  # Get the index of the selected row
    selected_room = df.iloc[selected_row_index]["Room"]  

    video_file = df.iloc[selected_row_index]["filepath"] 
    # **Use actual filepath from selection**
    if pd.notna(video_file) and os.path.exists(video_file):
        st.write(f"### Playing Video for {selected_room}")
        st.video(video_file)
    else:
        st.warning("⚠️ No video available for this selection.")







