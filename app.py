#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Photon Detection Event Analyzer Streamlit App

This Streamlit application processes binary TDC data files from quTAG devices,
extracts start-stop intervals between designated channels, performs extensive
data analysis including harmonic detection, and visualizes the results.
It is optimized for large datasets, automatically computes the total measurement
time from the data, and includes comprehensive logging for troubleshooting.
"""

import numpy as np
import pandas as pd
import logging
import streamlit as st
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import time
from threading import Thread
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ---------------------------
# Device Configuration
# ---------------------------

devices = {
    "quTAG_HR": {
        "description": "High Resolution variant of the quTAG time tagger",
        "timing_jitter_rms_ps": 2.4,             # 2.4 ps RMS jitter
        "min_pulse_separation_ps": 40e3,         # 40,000 ps = 40 ns
        "number_of_stop_channels": 16,           # Up to 16 stop channels
        "max_event_rate_per_channel_mCps": 25,   # 25 Mcps
        "sustained_throughput_rate_mCps": 100,   # 100 Mcps (USB3.0)
    },
    "quTAG": {  
        "description": "Standard variant of the quTAG time tagger",
        "timing_jitter_rms_ps": 3.0,  
        "min_pulse_separation_ps": 40e3,  
        "number_of_stop_channels": 4,  
        "max_event_rate_per_channel_mCps": 25,  
        "sustained_throughput_rate_mCps": 100,  
    },
    "quTAG_MC": {
        "description": "Multi Channel variant of the quTAG time tagger",
        "timing_jitter_rms_ps": 20,            # 20 ps RMS jitter
        "min_pulse_separation_ps": 10e3,        # 10,000 ps = 10 ns
        "number_of_stop_channels": 32,          # Up to 32 stop channels
        "max_event_rate_per_channel_mCps": 100,  # 100 Mcps
        "sustained_throughput_rate_mCps": 100,  # 100 Mcps (USB3.0)
    },
}

# ---------------------------
# Function Definitions
# ---------------------------

@st.cache_data(show_spinner=False)
def read_timetag_file(file_bytes, chunk_size=10**6):
    """
    Reads and parses the .timetag binary file efficiently.

    Parameters:
        file_bytes (bytes): Bytes of the uploaded file.
        chunk_size (int): Number of records to read per chunk.

    Returns:
        tuple: Two NumPy arrays containing timestamps and channel numbers.
    """
    header_size = 40  # bytes
    record_size = 10  # bytes

    timestamps = []
    channels = []
    min_timestamp = None
    max_timestamp = None

    f = BytesIO(file_bytes)
    header = f.read(header_size)
    if len(header) < header_size:
        st.error(f"Incomplete header. Expected {header_size} bytes, got {len(header)} bytes.")
        return None, None

    # Calculate total number of records
    f.seek(0, 2)  # Move to end of file
    file_size = f.tell()
    data_size = file_size - header_size
    total_records = data_size // record_size
    f.seek(header_size)  # Reset to position after header

    logging.info(f"Total records to read: {total_records}")

    # Read data in chunks
    for chunk_num in range(0, total_records, chunk_size):
        records_to_read = min(chunk_size, total_records - chunk_num)
        data = f.read(records_to_read * record_size)
        if len(data) != records_to_read * record_size:
            logging.warning(f"Expected {records_to_read * record_size} bytes, got {len(data)} bytes.")
            continue

        # Use NumPy's frombuffer for efficient unpacking
        try:
            dtype = np.dtype([('timestamp', '<Q'), ('channel', '<H')])
            data_array = np.frombuffer(data, dtype=dtype)
            timestamps_chunk = data_array['timestamp']
            channels_chunk = data_array['channel']
            timestamps.append(timestamps_chunk)
            channels.append(channels_chunk)

            # Update min and max timestamps
            chunk_min = timestamps_chunk.min()
            chunk_max = timestamps_chunk.max()
            if min_timestamp is None or chunk_min < min_timestamp:
                min_timestamp = chunk_min
            if max_timestamp is None or chunk_max > max_timestamp:
                max_timestamp = chunk_max

            logging.info(f"Processed chunk {chunk_num // chunk_size + 1}: {records_to_read} records.")
        except Exception as e:
            logging.error(f"NumPy unpacking failed: {e}")
            continue

    if timestamps and channels:
        timestamps = np.concatenate(timestamps)
        channels = np.concatenate(channels)
        return timestamps, channels
    else:
        return None, None


@st.cache_data(show_spinner=False)
def display_channel_distribution(channels):
    """
    Returns the distribution of events across channels.

    Parameters:
        channels (np.ndarray): Array of channel numbers.

    Returns:
        dict: Dictionary with channel numbers as keys and event counts as values.
    """
    unique_channels, counts = np.unique(channels, return_counts=True)
    channel_counts = dict(zip(unique_channels, counts))
    return channel_counts


@st.cache_data(show_spinner=False)
def find_start_stop_intervals(
    timestamps, 
    channels, 
    start_channel, 
    stop_channels, 
    min_pulse_separation_ps, 
    device_max_event_rate_mCps, 
    total_time_s
):
    """
    Finds start-stop intervals between start and stop channels.

    Parameters:
        timestamps (np.ndarray): Array of timestamps.
        channels (np.ndarray): Array of channel numbers.
        start_channel (int): Channel number designated as "start".
        stop_channels (list of int): List of channel numbers designated as "stop" channels.
        min_pulse_separation_ps (float): Minimum pulse separation in picoseconds.
        device_max_event_rate_mCps (float): Maximum event rate per channel in Mega counts per second.
        total_time_s (float): Total data collection time in seconds.

    Returns:
        tuple: 
            - np.ndarray: Array of start-stop intervals in picoseconds.
            - np.ndarray: Array of corresponding stop event timestamps in picoseconds.
            - dict: Dictionary of event rates per channel including start channel.
    """
    intervals = []
    stop_times_paired = []  # List to store corresponding stop timestamps
    event_rates = {}
    used_start_indices = set()  # To keep track of used start events

    # Extract start and stop events
    start_indices = np.where(channels == start_channel)[0]
    start_times = timestamps[start_indices]

    # Dictionary to hold stop indices per channel
    stop_times_dict = {}
    for stop_ch in stop_channels:
        stop_indices = np.where(channels == stop_ch)[0]
        stop_times = timestamps[stop_indices]
        stop_times_dict[stop_ch] = stop_times

        # Compute event rate for stop channel
        stop_event_rate = len(stop_times) / total_time_s  # counts per second
        event_rates[stop_ch] = {
            'rate_cps': stop_event_rate,
            'counts': len(stop_times)
        }

    # Compute event rate for start channel
    start_event_rate = len(start_times) / total_time_s  # counts per second
    event_rates[start_channel] = {
        'rate_cps': start_event_rate,
        'counts': len(start_times)
    }

    # Sort all stop events with their channels
    all_stops = []
    for stop_ch, stop_times in stop_times_dict.items():
        for st in stop_times:
            all_stops.append((st, stop_ch))
    all_stops.sort()  # Sort by timestamp

    # Initialize start pointer
    start_ptr = 0
    num_starts = len(start_times)

    for stop_time, stop_ch in all_stops:
        # Find the first start event that occurs before the stop and is not used
        while start_ptr < num_starts and start_times[start_ptr] + min_pulse_separation_ps <= stop_time:
            if start_ptr not in used_start_indices:
                interval = stop_time - start_times[start_ptr]
                intervals.append(interval)
                stop_times_paired.append(stop_time)  # Associate stop time with interval
                used_start_indices.add(start_ptr)
                start_ptr += 1
                break
            start_ptr += 1
        else:
            # No available start event found for this stop
            continue

    return np.array(intervals), np.array(stop_times_paired), event_rates


@st.cache_data(show_spinner=False)
def outlier_detection(intervals, z_threshold=3.0):
    """
    Detects and removes outliers using Z-scores.

    Parameters:
        intervals (np.ndarray): Array of start-stop intervals in picoseconds.
        z_threshold (float): Z-score threshold for outlier detection.

    Returns:
        np.ndarray: Array of intervals with outliers removed.
    """
    if len(intervals) == 0:
        logging.warning("No intervals available for outlier detection.")
        return intervals  # Early exit as per code review comment

    mean = np.mean(intervals)
    std_dev = np.std(intervals)
    if std_dev == 0:
        logging.warning("Standard deviation is zero. No outliers to remove.")
        return intervals

    z_scores = np.abs((intervals - mean) / std_dev)
    cleaned_intervals = intervals[z_scores <= z_threshold]  # Remove outliers beyond threshold

    logging.info(f"Removed {len(intervals) - len(cleaned_intervals)} outliers from data.")
    return cleaned_intervals


@st.cache_data(show_spinner=False)
def calculate_normalization(start_events, stop_events, bin_width, total_time_s):
    """
    Calculates the normalization factor based on the provided formula:
    N = R_start * R_stop * W * T_total

    Parameters:
        start_events (int): Number of start events.
        stop_events (int): Number of stop events.
        bin_width (int): Width of each histogram bin in picoseconds.
        total_time_s (float): Total data collection time in seconds.

    Returns:
        float: Normalization factor.
    """
    R_start = start_events / total_time_s  # Events per second
    R_stop = stop_events / total_time_s    # Events per second
    W = bin_width * 1e-12  # bin_width in seconds

    normalization_factor = R_start * R_stop * W * total_time_s
    logging.info(f"Normalization Factor: {normalization_factor:.6e}")
    return normalization_factor


def export_intervals_to_csv(intervals, output_filename='intervals.csv'):
    """
    Exports start-stop intervals to a CSV file.

    Parameters:
        intervals (np.ndarray): Array of start-stop intervals in picoseconds.
        output_filename (str): Name of the output CSV file.

    Returns:
        BytesIO: BytesIO object of the CSV file for download.
    """
    try:
        df = pd.DataFrame({'Start_Stop_Interval_ps': intervals})
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        logging.info(f"Intervals successfully exported to '{output_filename}'.")
        return csv_buffer
    except Exception as e:
        logging.error(f"An error occurred while exporting intervals to CSV: {e}")
        return None

# ---------------------------
# Streamlit App Layout
# ---------------------------

st.set_page_config(page_title="Photon Detection Event Analyzer", layout="wide")
st.title("🔬 Photon Detection Event Analyzer")

st.markdown("""
This application processes binary TDC data files from quTAG devices, extracts start-stop intervals between designated channels,
performs extensive data analysis including harmonic detection, and visualizes the results. It is optimized for large datasets,
automatically computes the total measurement time from the data, and includes comprehensive logging for troubleshooting.
""")

# Sidebar for user inputs
st.sidebar.header("📁 Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload .timetag File", type=["timetag", "bin", "dat"])

if uploaded_file is not None:
    st.sidebar.success("✅ File uploaded successfully.")
    file_bytes = uploaded_file.read()
else:
    st.sidebar.warning("⚠️ Please upload a .timetag binary data file to proceed.")
    st.stop()

# Device selection
device_options = list(devices.keys())
device = st.sidebar.selectbox("Select Device", options=device_options, index=0)
selected_device = devices[device]
st.sidebar.write(f"**Device Description:** {selected_device['description']}")

# Channel selection
with st.sidebar.expander("🔧 Configure Channels"):
    start_channel = st.number_input(
        "Start Channel", 
        min_value=0, 
        max_value=selected_device["number_of_stop_channels"]*2, 
        value=1, 
        step=1
    )
    stop_channels = st.multiselect(
        "Stop Channels",
        options=list(range(0, selected_device["number_of_stop_channels"]*2)),
        default=[2],
        help=f"Select up to {selected_device['number_of_stop_channels']} unique stop channels."
    )
    if len(stop_channels) > selected_device["number_of_stop_channels"]:
        st.error(f"Number of stop channels exceeds the device's maximum ({selected_device['number_of_stop_channels']}).")
    # Ensure unique stop channels
    if len(stop_channels) != len(set(stop_channels)):
        st.error("❌ Duplicate stop channels detected. Each stop channel must be unique.")

# Histogram parameters
with st.sidebar.expander("📊 Histogram Parameters"):
    bin_size = st.number_input(
        "Bin Size (ps)", 
        min_value=1, 
        max_value=1_000_000, 
        value=1000, 
        step=100
    )
    bin_range_input = st.text_input(
        "Bin Range (min,max in ps)", 
        value="0,10000", 
        help="Enter two comma-separated values, e.g., 0,10000"
    )
    try:
        bin_range = tuple(map(int, bin_range_input.split(',')))
        if len(bin_range) != 2:
            st.error("❌ Bin range must have exactly two values: min and max.")
            bin_range = None
    except:
        st.error("❌ Invalid bin range format. Please enter two integers separated by a comma.")
        bin_range = None

    normalize = st.checkbox("📏 Normalize Histogram", value=False)

# Outlier removal settings
with st.sidebar.expander("🎛️ Outlier Removal Settings"):
    remove_outliers = st.checkbox("🗑️ Enable Outlier Removal", value=False)
    z_threshold = st.slider(
        "⚖️ Z-score Threshold", 
        min_value=1.0, 
        max_value=5.0, 
        value=3.0, 
        step=0.1, 
        help="Set the Z-score threshold for outlier detection."
    )

# Export options
export_csv = st.sidebar.checkbox("💾 Export Intervals to CSV", value=False)

# Output filenames
with st.sidebar.expander("💾 Output Settings"):
    output_filename = st.text_input("Histogram Output Filename", value="histogram.png")
    csv_filename = st.text_input("CSV Output Filename", value="intervals.csv")

# Run Analysis Button
run_analysis = st.sidebar.button("▶️ Run Analysis")

if run_analysis:
    st.header("📊 Analysis Results")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Validate stop channels
    if len(stop_channels) > selected_device["number_of_stop_channels"]:
        st.error(f"❌ Number of stop channels provided ({len(stop_channels)}) exceeds the device's maximum ({selected_device['number_of_stop_channels']}).")
        st.stop()

    try:
        if len(stop_channels) != len(set(stop_channels)):
            raise ValueError("Duplicate stop channels detected. Each stop channel must be unique.")
    except ValueError as ve:
        st.error(f"❌ {ve}")
        st.stop()

    # Read and parse the .timetag file with caching
    with st.spinner("📥 Reading and parsing the .timetag file..."):
        timestamps, channels = read_timetag_file(file_bytes, chunk_size=10**6)

    if timestamps is None or channels is None:
        st.error("❌ Failed to parse the .timetag file. Please ensure it is in the correct format.")
        st.stop()

    st.success(f"✅ **Total events parsed:** {len(timestamps)}")

    # Compute total measurement time
    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()
    total_time_ps = max_timestamp - min_timestamp  # Assuming timestamps are in picoseconds
    if total_time_ps <= 0 or len(timestamps) < 2:
        st.error("❌ Invalid total measurement time calculated. Ensure that timestamps are in increasing order and sufficient data is present.")
        st.stop()
    total_time_s = total_time_ps * 1e-12  # Convert picoseconds to seconds
    st.info(f"🕒 **Total Measurement Time:** {total_time_s:.6f} seconds (Computed from data)")

    # Convert timestamps to relative time in seconds
    relative_timestamps = (timestamps - min_timestamp) * 1e-12  # Convert ps to seconds

    # Display channel distribution
    with st.spinner("📊 Computing channel distribution..."):
        channel_distribution = display_channel_distribution(channels)
    st.subheader("📊 Channel Distribution")
    df_channel_dist = pd.DataFrame([
        {"Channel": ch, "Event Count": cnt}
        for ch, cnt in sorted(channel_distribution.items())
    ])
    st.dataframe(df_channel_dist.style.format({"Event Count": "{:,.0f}"}))

    # Plot channel distribution using Plotly
    st.subheader("📈 Channel Distribution Plot")
    fig_channel_dist = px.bar(
        df_channel_dist,
        x='Channel',
        y='Event Count',
        labels={'Channel': 'Channel Number', 'Event Count': 'Number of Events'},
        title='Distribution of Events Across Channels',
        color='Event Count',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_channel_dist, use_container_width=True)

    # Plot event timeline over the entire dataset using Plotly
    st.subheader("🕒 Event Timeline")

    # Automatic downsampling
    MAX_TIMELINE_POINTS = 200000  # Adjust as needed
    if len(relative_timestamps) > MAX_TIMELINE_POINTS:
        sample_indices = np.random.choice(len(relative_timestamps), MAX_TIMELINE_POINTS, replace=False)
        plot_timestamps = relative_timestamps[sample_indices]
        plot_channels = channels[sample_indices]
    else:
        plot_timestamps = relative_timestamps
        plot_channels = channels

    # Plot the event timeline with larger markers
    with st.spinner("📈 Generating Event Timeline plot..."):
        fig_event_timeline = px.scatter(
            x=plot_timestamps,
            y=plot_channels,
            labels={'x': 'Relative Timestamp (s)', 'y': 'Channel'},
            title='Event Timeline',
            color=plot_channels,
            color_continuous_scale='RdBu',
            opacity=0.6,
            hover_data={'Relative Timestamp (s)': plot_timestamps, 'Channel': plot_channels}
        )
        fig_event_timeline.update_traces(marker=dict(size=4))  # Increased marker size
        st.plotly_chart(fig_event_timeline, use_container_width=True)

    # Find start-stop intervals and monitor event rates
    with st.spinner("🔍 Identifying start-stop intervals..."):
        intervals, stop_times_paired, event_rates = find_start_stop_intervals(
            timestamps,
            channels,
            start_channel,
            stop_channels,
            selected_device["min_pulse_separation_ps"],
            selected_device["max_event_rate_per_channel_mCps"],
            total_time_s
        )

    st.subheader("📏 Start-Stop Intervals")
    st.write(f"**Total Start Events:** {event_rates[start_channel]['counts']}")
    total_stop_events = sum([details['counts'] for ch, details in event_rates.items() if ch != start_channel])
    st.write(f"**Total Stop Events:** {total_stop_events}")
    st.write(f"**Total Paired Intervals Found:** {len(intervals)}")

    # Display event rates in a table
    st.write("**📈 Event Rates per Channel:**")
    df_event_rates = pd.DataFrame([
        {
            "Channel": ch,
            "Event Rate (cps)": f"{details['rate_cps']:.2e}",
            "Number of Events": details['counts']
        }
        for ch, details in sorted(event_rates.items())
    ])
    st.table(df_event_rates)

    # Display warnings for high event rates
    for ch, details in event_rates.items():
        rate_cps = details['rate_cps']
        if ch == start_channel:
            max_rate_cps = selected_device["max_event_rate_per_channel_mCps"] * 1e6
            if rate_cps > max_rate_cps:
                st.warning(f"⚠️ **Start Channel {ch}** event rate {rate_cps:.2e} cps exceeds maximum {max_rate_cps:.2e} cps.")
        else:
            max_rate_cps = selected_device["max_event_rate_per_channel_mCps"] * 1e6
            if rate_cps > max_rate_cps:
                st.warning(f"⚠️ **Stop Channel {ch}** event rate {rate_cps:.2e} cps exceeds maximum {max_rate_cps:.2e} cps.")

    # Check if any intervals were found
    if len(intervals) == 0:
        st.error("❌ No start-stop intervals found. Please verify your start and stop channels.")
        st.stop()

    # Detect and remove outliers using Z-scores (if enabled)
    cleaned_intervals = intervals
    cleaned_stop_times = stop_times_paired

    if remove_outliers:
        with st.spinner("🗑️ Removing outliers from intervals..."):
            cleaned_intervals = outlier_detection(intervals, z_threshold=z_threshold)
            # Create a mask based on which intervals were kept
            mean = np.mean(intervals)
            std_dev = np.std(intervals)
            if std_dev == 0:
                logging.warning("Standard deviation is zero. No outliers to remove.")
                mask = np.ones(len(intervals), dtype=bool)
            else:
                z_scores = np.abs((intervals - mean) / std_dev)
                mask = z_scores <= z_threshold
                st.info(f"🗑️ Removed {len(intervals) - np.sum(mask)} outliers from data based on Z-score threshold of {z_threshold}.")
            cleaned_stop_times = stop_times_paired[mask]
    else:
        st.info("🔍 Outlier detection not enabled. Proceeding without removing outliers.")

    # Calculate normalization factor if normalization is enabled
    normalization_factor = None
    if normalize:
        stop_events = sum([details['counts'] for ch, details in event_rates.items() if ch != start_channel])
        start_events = event_rates[start_channel]['counts']
        if start_events == 0 or stop_events == 0:
            st.error("❌ Start or Stop events count is zero. Cannot perform normalization.")
            st.stop()
        normalization_factor = calculate_normalization(
            start_events,
            stop_events,
            bin_size,
            total_time_s
        )
    else:
        st.info("📉 Normalization not enabled. Proceeding without normalization.")

    # Plot interval statistics using Plotly
    st.subheader("📊 Interval Statistics")
    if len(cleaned_intervals) > 0:
        mean_interval = np.mean(cleaned_intervals) * 1e-3  # Convert to nanoseconds
        median_interval = np.median(cleaned_intervals) * 1e-3  # Convert to nanoseconds
        std_interval = np.std(cleaned_intervals) * 1e-3  # Convert to nanoseconds
        st.write(f"**Mean Interval:** {mean_interval:.2f} ns")
        st.write(f"**Median Interval:** {median_interval:.2f} ns")
        st.write(f"**Standard Deviation:** {std_interval:.2f} ns")
    else:
        st.warning("❗ No intervals available for statistical analysis.")

    # Plot Interval Distribution without Statistics on the Plot
    st.subheader("📈 Start-Stop Interval Distribution")
    if len(cleaned_intervals) > 0:
        with st.spinner("📈 Generating Interval Distribution plot..."):
            fig_distribution = px.histogram(
                cleaned_intervals,  # Keep intervals in picoseconds
                nbins=50,  # Adjusted for clarity
                labels={'value': 'Interval Length (ps)'},
                title='Start-Stop Interval Distribution',
                color_discrete_sequence=['#1f77b4'],  # A distinct blue color
                opacity=0.75,
                marginal="box"  # Adds a box plot for summary statistics
            )
            if normalize and normalization_factor is not None:
                fig_distribution.update_layout(yaxis_title='Normalized Counts')
            else:
                fig_distribution.update_layout(yaxis_title='Counts')
            
            # Enhance layout for better aesthetics
            fig_distribution.update_layout(
                title={
                    'text': 'Start-Stop Interval Distribution',
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                xaxis_title='Interval Length (ps)',
                yaxis_title='Counts',
                template='plotly_white',
                bargap=0.1,  # Slightly increased gap for better separation
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_distribution, use_container_width=True)
    else:
        st.warning("❗ No intervals available to plot distribution.")

    # Function to plot paired events scatter plot
    def plot_paired_scatter():
        with st.spinner("📈 Generating Paired Events Scatter Plot..."):
            fig_paired = go.Figure(
                data=go.Scattergl(
                    x=cleaned_stop_times * 1e-12,  # Convert to seconds
                    y=cleaned_intervals,           # Keep intervals in picoseconds
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=cleaned_intervals,   # Color by interval in ps
                        colorscale='Viridis',
                        showscale=True,
                        opacity=0.6
                    ),
                    hoverinfo='x+y'
                )
            )
            fig_paired.update_layout(
                title='Paired Start-Stop Intervals Over Time',
                xaxis_title='Stop Event Relative Timestamp (s)',
                yaxis_title='Interval Length (ps)',
                template='plotly_white'
            )
            st.plotly_chart(fig_paired, use_container_width=True)

    # Function to plot hexbin plot
    def plot_paired_hexbin():
        with st.spinner("📈 Generating Paired Events Hexbin Plot..."):
            fig_hex = px.density_heatmap(
                x=cleaned_stop_times * 1e-12,  # Convert to seconds
                y=cleaned_intervals,           # Keep intervals in picoseconds
                nbinsx=100,
                nbinsy=100,
                labels={'x': 'Stop Event Relative Timestamp (s)', 'y': 'Interval Length (ps)'},
                title='Paired Start-Stop Intervals Hexbin Density',
                color_continuous_scale='Viridis'
            )
            fig_hex.update_layout(
                template='plotly_white'
            )
            st.plotly_chart(fig_hex, use_container_width=True)

    # Implement timeout mechanism using ThreadPoolExecutor
    def plot_with_timeout():
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(plot_paired_scatter)
        try:
            future.result(timeout=15)  # 15 seconds timeout
        except Exception:
            st.warning("⚠️ Paired Events Scatter Plot is taking too long to load. Displaying Hexbin Plot instead.")
            plot_paired_hexbin()

    # Run the plot with timeout
    plot_with_timeout()

    # Plot interval variation over time using Plotly
    st.subheader("📉 Interval Variation Over Time")
    MAX_VARIATION_POINTS = 50000  # Adjust as needed for performance
    if len(cleaned_intervals) > MAX_VARIATION_POINTS:
        sample_indices_var = np.random.choice(len(cleaned_intervals), MAX_VARIATION_POINTS, replace=False)
        plot_stop_times_var = cleaned_stop_times[sample_indices_var] * 1e-12  # Convert to seconds
        plot_intervals_var = cleaned_intervals[sample_indices_var]           # Keep in picoseconds
    else:
        plot_stop_times_var = cleaned_stop_times * 1e-12  # Convert to seconds
        plot_intervals_var = cleaned_intervals           # Keep in picoseconds

    if len(plot_intervals_var) > 0:
        with st.spinner("📈 Generating Interval Variation Over Time plot..."):
            fig_variation = go.Figure(
                data=go.Scattergl(
                    x=plot_stop_times_var,
                    y=plot_intervals_var,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=plot_intervals_var,    # Color by interval in ps
                        colorscale='Plasma',
                        showscale=True,
                        opacity=0.6
                    ),
                    hoverinfo='x+y'
                )
            )
            fig_variation.update_layout(
                title='Variation of Start-Stop Intervals Over Time',
                xaxis_title='Stop Event Relative Timestamp (s)',
                yaxis_title='Interval Length (ps)',
                template='plotly_white'
            )
            st.plotly_chart(fig_variation, use_container_width=True)
    else:
        st.warning("❗ Insufficient data to plot interval variation over time.")

    # Export intervals to CSV (optional)
    if export_csv:
        st.subheader("💾 Export Intervals to CSV")
        with st.spinner("💾 Preparing CSV for download..."):
            csv_buffer = export_intervals_to_csv(cleaned_intervals, output_filename=csv_filename)
        if csv_buffer:
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer,
                file_name=csv_filename,
                mime='text/csv'
            )
        else:
            st.error("❌ Failed to export CSV.")
    else:
        st.info("📂 CSV export not requested. Skipping export.")

    st.success("✅ **Analysis Completed Successfully!**")

else:
    st.info("🔍 Configure the parameters and click '▶️ Run Analysis' to start processing.")

# ---------------------------
# Footer
# ---------------------------

st.markdown("""
---
**Photon Detection Event Analyzer** | Developed by Jan FitzGibbon for NuQuantum ⚛️
""")
