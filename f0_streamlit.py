import pandas as pd
import streamlit as st
import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import f0_streamlit_analysis as f0_analysis
from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype
import json

st.set_page_config(layout="wide")

# Initialize session state for triggering the graph
if "find_clips" not in st.session_state:
    st.session_state.find_clips = False

if "sfig" not in st.session_state:
    st.session_state.sfig= None 

if "cdf" not in st.session_state:
    st.session_state.cdf = None  

if "sdf" not in st.session_state:
    st.session_state.sdf = None  

if "selected_room" not in st.session_state:
    st.session_state.selected_room = None  # Start with no selection


BASE_DIR = "/home/alistairfraser/data/buckets/oregon.birdconv.mp4/tweety/"
# BASE_DIR = "/Users/alistair/code/BirdCallAuth/test"

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
                    # print(f"Found {mp4_files[0]} for {room}")

                    filename = os.path.basename(mp4_files[0]).replace(".mp4", "")
                    parts = filename.split("-")
                    
                    if len(parts) > 1:  # Ensure there are names after the timestamp
                        extracted_names = parts[1:]  # All elements after the timestamp

                        # Assign `name_index` based on position in the filename
                        for idx, name in enumerate(extracted_names):
                            df.loc[ (df["room_name"] == room) & (df["name"] == name) , "name_index"] = idx

                    f0_data = os.path.join(room_dir, os.path.splitext(mp4_files[0])[0] + "_f0.json")
                    if os.path.exists(f0_data):
                        # print(f"Found {f0_data}")
                        df.loc[ df["room_name"] == room, "f0" ] = f0_data
                    else :
                        print(f"Could not find {f0_data}")
    return df

# Load and cache the dataset
df = load_data()


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
st.sidebar.header("Birdconv Data Explorer")

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
        "Automatic Gain Control", df["agc_enabled"].unique(), default=df["agc_enabled"].unique()
    )
    df = df[df["agc_enabled"].isin(agc_enabled_selected)]


if not df.empty:
    # `bypass_voice_processing` filter (category dropdown)
    bypass_voice_processing_selected = st.sidebar.multiselect(
        "Bypass Voice Processing", df["bypass_voice_processing"].unique(), default=df["bypass_voice_processing"].unique()
    )
    df = df[df["bypass_voice_processing"].isin(bypass_voice_processing_selected)]


st.sidebar.markdown("## F0 processing settings:")  

cutoff = st.sidebar.slider(
    "Cage Noise frequency cutoffs (hz)",
    min_value=0, max_value=5000, value=(1700, 3000), step=50,
    help = "bird calls are within this range, cage noise is above and below this range"
)

# cutoff            = st.sidebar.slider("Cage Noise frequency cutoff (hz)", min_value=0.0, max_value=5000.0, value=1700.0, step=5.0, help="f0 values below this frequency are considered cage noise")
segment_length    = st.sidebar.slider("Segment length (seconds)", min_value=0.1, max_value=60.0, value=10.0, step=1.0,help="length of each segment for f0 calculations, longer segements = faster calculations, shorter segments = slower calculations. ")   
cluster_penalty   = st.sidebar.slider("Cluster penalty ", min_value=0.0, max_value=10.0, value=2.0, step=1.0,help="cluster penalty determines how uniform the clips are in activity and balance. (0 generates more clips, 10 generates fewer clips)")
include_cage      = st.sidebar.checkbox("Cage noise subtitles", value=True,help="when watching a video, show subtitles for cage noise")


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
    "f0",
    "name_index"
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


# Allow user to select a row
st.markdown("##### Conversations")
event = st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_config, 
    on_select="rerun",
    selection_mode="single-row")

# Display total duration and unique rooms
st.markdown(f"{num_unique_rooms} conversations. {formatted_total_duration}")
st.write("---")

# Get the selected rows

# Example usage in Streamlit:
# (Make sure to replace these file paths with valid paths on your system.)


if st.button("Find Songs"):
    segments_df = f0_analysis.collect_song_segments(df)
    
if segments_df is not None:
    st.markdown("##### Song Segments")
    
    # Configure column settings
    column_config = {
        "Start": st.column_config.TextColumn("Start Time"),
        "Duration": st.column_config.TextColumn("Duration"),
        "filepath": {"hidden": True},
        "start_time": {"hidden": True},
        "end_time": {"hidden": True},
        "f0": {"hidden": True}
    }
    
    # Display segments table with selection
    segment_event = st.dataframe(
        segments_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row"
    )
    
else:
    st.warning("No song segments found in the selected files.")

# Play video when segment is selected
if segment_event and segment_event.selection.rows:
    f0_analysis.play_video(segment_event, segments_df, cutoff, include_cage)

if st.button("Download json files..."):
    zip_data = f0_analysis.create_zip( df['f0'].dropna().unique().tolist() )
    if zip_data is not None:
        st.write("files ready...")
        st.download_button(
            label="Download ZIP",
            data=zip_data,
            file_name="files.zip",
            mime="application/zip"
        )
    else:
        st.write("No files selected.")

f0_analysis.play_video(event,df,cutoff,include_cage)

loading_progress_bar   = st.progress(0)
loading_progress_text  = st.empty()
# st.write("---")
st.markdown("##### F0 Analysis")
f0 = f0_analysis.F0Analysis( df , cutoff, segment_length, cluster_penalty,progress_bar=loading_progress_bar, progress_text=loading_progress_text)
if loading_progress_bar is not None and loading_progress_text is not None:
    loading_progress_bar.progress(100)
    loading_progress_text.write("")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"###### {segment_length} second segments by category ")
    f0.plot_pie_chart()
with col2:
    st.markdown("###### Number of segments by bird sound %")
    f0.plot_stacked_bar_chart()
with col3:
    st.markdown("###### F0 frequency distribution ")
    f0.plot_combined_histogram()

col1 = st.columns(1)[0]

progress_bar = None
progess_text = None
with col1:

    column_config = {
        "Call %": st.column_config.NumberColumn("Call %", format="%.1f%%"),
        "Cage Noise %": st.column_config.NumberColumn("Cage Noise %", format="%.1f%%")
    }
    st.dataframe(f0.bdf, column_config=column_config, use_container_width=True, hide_index=True)
    st.write("---")
    st.markdown("##### Conversation clips")
    st.write("Cluster to find clips categorized by Call percentage and conversation dominance/balance score")
    if st.button("Find Clips", help="Adjust cluster penalty for shorter or longer clips. Increase segment length to speed up calculation.") : 
        st.session_state.find_clips = True  # Set session state to True
        progress_bar   = st.progress(0)
        progress_text  = st.empty()
        progress_text.write(f"Finding clips for { len(f0.filenames) } files. Please wait...")

col1, col2 = st.columns(2)

with col1 : 
    if st.session_state.find_clips:
        st.session_state.find_clips = False  # Reset session state to False
        st.session_state.sfig, st.session_state.cdf, st.session_state.sdf = f0.plot_2d_heatmap(progress_bar=progress_bar, progress_text=progress_text)

        if progress_bar is not None and progress_text is not None:
            progress_bar.progress(100)
            progress_text.write("")

    if st.session_state.sfig is not None:
        st.markdown("###### Dominance Score  vs Bird Sound Activity Level heatmap ")
        st.pyplot(st.session_state.sfig)

with col2:
    if st.session_state.cdf is not None:

        # Get min and max values from dataframe
        Call_min_value = st.session_state.cdf["Call Percentage"].min()
        Call_max_value = st.session_state.cdf["Call Percentage"].max()
        balance_min_value = st.session_state.cdf["Dominance Score"].min()
        balance_max_value = st.session_state.cdf["Dominance Score"].max()

        st.markdown("###### Filter Clips")

        # Dual slider for filtering
        Call_min, Call_max = st.slider(
            "Call Percentage",
            min_value=Call_min_value, max_value=Call_max_value, value=(Call_min_value, Call_max_value), format="%.1f%%",
            help = " Show clips with a Call percentage between the selected range"
        )
        balance_min, balance_max = st.slider(
            "Dominance Score (e.g. 0=balanced, 1=single bird dominant )",
            min_value=balance_min_value, max_value=balance_max_value, value=(balance_min_value, balance_max_value), format="%.1f%%", 
            help = " Show clips with a Call percentage between the selected range"
        )

if st.session_state.cdf is not None:
    # Apply filtering
    filtered_df = st.session_state.cdf[
        (st.session_state.cdf["Call Percentage"].between(Call_min, Call_max)) &
        (st.session_state.cdf["Dominance Score"].between(balance_min, balance_max))
    ]


    # Define percentage formatting for columns
    column_config = {
        "Call Percentage": st.column_config.NumberColumn("Call %", format="%.1f%%"),
        "Dominance Score": st.column_config.NumberColumn("Dominance Score", format="%.1f%%")
    }

    # Define columns to hide
    columns_to_hide = ["filepath", "start_time", "end_time", "duration","f0"]

    # Merge both configurations
    for col in columns_to_hide:
        column_config[col] = {"hidden": True}  # Keep existing formatting

    # Display DataFrame with both percentage formatting and hidden columns

    st.markdown("###### Filtered Clips")
    event = st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row"
    )
    if event and event.selection.rows:
        st.markdown("###### Selected File : Call Percentage and Dominance Score vs Time")
        f0_analysis.plot_line_chart(event,filtered_df,st.session_state.cdf,st.session_state.sdf)
        f0.plot_diagnostics(event,filtered_df)

    f0_analysis.play_video(event,filtered_df,cutoff,include_cage)

    if event and event.selection.rows:
        audio_analyis = f0_analysis.AudioAnalysis(event,filtered_df,cutoff)
        audio_analyis.audio_plot_interface(clip_start_time=filtered_df.iloc[event.selection.rows[0]].get( "start_time", 0))
       
