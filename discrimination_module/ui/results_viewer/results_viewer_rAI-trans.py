import sys
import re
import SimpleITK as sitk
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

# Custom
from image_utils import *
from visualization import *

import streamlit as st
import json

from typing import *
import logging
import pydantic
from rich.logging import RichHandler
from rich.traceback import install

install()


class Configuration(pydantic.BaseModel):
    # Use ClassVar to mark class variables
    config_dir: ClassVar[Path] = Path(".config")  # Hidden folder to store states
    mapper: ClassVar[Dict[str, str]] = {
        'Image Directory': 'IMAGE_DIR',
        'Attention Map Directory': 'ATTN_DIR',
        'Segmentation Directory': 'SEGMENTATION_DIR',
        'Prediction CSV Directory': 'CSV_DIR',
        'ID Globber Regex': 'ID_GLOBBER',
        'Output Directory': 'OUTPUT_DIR'
    }
    GRID_COLS: ClassVar[int] = 5
    DEFAULT_WINDOW_RANGE: ClassVar[Tuple[int, int]] = (25, 99)
    DEFAULT_ATTN_THRESHOLD: ClassVar[Tuple[int, int]] = (15, 55)
    DEFAULT_OPACITY: ClassVar[float] = 0.5
    DEFAULT_CONTOUR_ALPHA: ClassVar[float] = 0.8
    DEFAULT_CONTOUR_WIDTH: ClassVar[int] = 2
    DEFAULT_SLICE_OFFSET: ClassVar[int] = 0
    ATTN_MIN_VALUE: ClassVar[int] = 0
    ATTN_MAX_VALUE: ClassVar[int] = 255
    HIST_LOWER_PERCENTILE: ClassVar[int] = 2
    HIST_UPPER_PERCENTILE: ClassVar[int] = 98

    # File paths - instance variables
    IMAGE_DIR: str = '.'
    ATTN_DIR: str = ''
    SEGMENTATION_DIR: str = ''
    CSV_DIR: str = 'results.csv'
    ID_GLOBBER: str = r"\w{0,5}\d+"
    OUTPUT_DIR: str = './saved_images'
    
    # Display settings - instance variables
    image_window_range: Tuple[int, int] = (25, 99)
    attn_threshold: Tuple[int, int] = (15, 55)
    attn_opacity: float = 0.5
    contour_alpha: float = 0.8
    contour_width: int = 2
    slice_offset: int = 0

    def save_to_json(self, filename: str):
        """Save the configuration to a JSON file."""
        try:
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.model_dump(), f, indent=2)
                logger.info(f"Configuration saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    @classmethod
    def load_from_json(cls, filename: str) -> 'Configuration':
        """Load the configuration from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                return cls(**data)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return cls()
    
    @classmethod
    def get_state_filename(cls, image_path: str) -> str:
        """Generate state filename based on image path stem and current date."""
        p = Path(image_path) if image_path else Path("default/default")
        stem = f"{p.parent.stem}_{p.stem}"
        timestamp = datetime.now().strftime("%Y%m%d%H")
        return f"{stem}_{timestamp}.json"
    
    @classmethod
    def list_available_states(cls) -> List[Path]:
        """List all available state files in the config directory."""
        if not cls.config_dir.exists():
            return []
        return sorted(cls.config_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    @classmethod
    def get_latest_state(cls) -> Optional[Path]:
        """Get the most recently modified state file."""
        states = cls.list_available_states()
        return states[0] if states else None
    
    @classmethod
    def save_state(cls, config: 'Configuration', image_path: str = None):
        """Save state to .config folder with generated filename."""
        cls.config_dir.mkdir(exist_ok=True)
        if image_path:
            filename = cls.get_state_filename(image_path)
        else:
            filename = cls.get_state_filename("default")
        filepath = cls.config_dir / filename
        config.save_to_json(str(filepath))
        return filepath


st.set_page_config(layout="wide")
st.write("# NPC Screening Results View")
st.write(
    """
    This is a small UI that is used to view the prediction results and the self attention the transformer is giving.
    You should have the results prepared in the  

    # Results Viewer (results_viewer_rAI-trans.py) Overview

    This Streamlit application is an interactive tool for visualizing and analyzing rAIdiologist predictions and transformer self-attention mechanisms.

    ## Main Features

    - **Image and Attention Map Visualization**: Displays medical images with overlaid transformer attention maps
    - **Prediction Analysis**: Shows model prediction outputs compared to ground truth labels
    - **Filtering Options**: Filter cases by true positives (TP), true negatives (TN), false positives (FP), or false negatives (FN)
    - **Flexible Attention Visualization**:
      - Select specific attention heads
      - Use maximum or average view across heads
      - Adjust window range and attention thresholds
      - Control attention map transparency
    """)
with st.expander("📃 Technical Details"):
    st.write(
        """

        # 📃 Technical Details

        - **Data Loading**: Loads image and attention map pairs from specified directories
        - **Statistical Analysis**: Displays metrics like accuracy, sensitivity, and specificity
        - **Interactive Elements**:
          - Intuitive UI controls for display settings
          - Navigation buttons for browsing cases
          - Histogram display of attention value distribution
          - Image saving functionality

        ## 📁 File Structure Example

        ```
        project/
        ├── results_viewer_rAI-trans.py    # Main application file
        ├── data/
        │   ├── images/                    # Original medical images
        │   ├── attention_maps/            # Generated attention maps
        │   └── predictions.csv            # Model prediction results
        ├── output/
        │   └── saved_visualizations/      # Saved overlay images
        └── utils/
            └── image_processing.py        # Helper functions for image manipulation
        ```

        ⚠️ Format of attention maps should be {ID}_pb_pred.nii.gz. The attention map can be in the same folder as the images
        or a separate folder.

        ## 🖥️ Display Settings

        | Category | Feature | Description |
        |----------|---------|-------------|
        | **Display Settings** | **Window Level Controls** | |
        | | Window Width | Controls the contrast range |
        | | Window Center | Controls the brightness midpoint |
        | | **Attention Map Settings** | |
        | | Opacity | Controls the transparency of the attention overlay |
        | | Threshold | Sets minimum value for attention to be displayed |
        | | Colormap | Changes the color scheme of the attention visualization |
        | | Attention Head Selection | Choose which transformer attention head to display |
        | | **Display Mode Options** | |
        | | Raw Image | Shows only the original medical image |
        | | Attention Only | Shows only the attention map |
        | | Overlay | Combines both with adjustable parameters |
        | **Visualizations** | **Attention Histogram** | |
        | | X-axis | Attention intensity values |
        | | Y-axis | Frequency/count of pixels at each intensity |
        | | Vertical line | Current threshold setting |
        | | **Performance Metrics Panel** | |
        | | Confusion Matrix | Visualization of model prediction results |
        | | ROC Curve | Shows model performance |
        | | Precision-Recall Curve | Evaluates model precision |
        | | **Comparison View** | |
        | | Original Image | Shows unprocessed image |
        | | Attention Map | Shows attention distribution |
        | | Overlaid Result | Shows combined view |
        
        ## 📊 Prediction results format guide
        
        The prediction results should be a .csv or .xlsx file with the following columns:
        1. Index: this needs to be the left-most column
        2. Decision_0: this is the AI-prediction reduced to binary prediction
        3. Prob_0: this is the original probability
        4. Truth_0 (Optional): this is the reference standard used for evaluation
        
        """

    )

# -- inistilize states
st.session_state['last_confirmation'] = st.session_state.get("last_confirmation", False)


# -- Add rich handler if it's not already there:
def setup_logger(logger):
    # Check if the RichHandler is already added
    if not any(isinstance(handler, RichHandler) for handler in logger.handlers):
        # Remove all existing handlers
        for handler in logger.handlers:
            logger.removeHandler(handler)

        # Add RichHandler if it's not present
        rich_handler = RichHandler(console=False, rich_tracebacks=True, tracebacks_show_locals=True,
                                   locals_max_length=20)
        logger.addHandler(rich_handler)
        logger.setLevel(logging.DEBUG)  # Set the logging level if needed

        # Log a test message
        logger.info("Logger setup complete with RichHandler.")

    return logger


# * Adding this handler to streamlit
# First remove the error message in streamlit by default
logger = st.logger.get_logger("streamlit.error_util")
for handler in logger.handlers:
    logger.removeHandler(handler)
# Setup the logger
logger = st.logger.get_logger("App")
setup_logger(logger)


# Introduce my own error handling
def set_global_exception_handler(f):
    import sys
    error_util = sys.modules["streamlit.error_util"]
    error_util.handle_uncaught_app_exception = f


set_global_exception_handler(logger.error)


def _exception_hook(exctype, value, traceback):
    """Custom exception hook for logging uncaught exceptions."""
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, traceback)
        return
    logger.error("Uncaught exception", exc_info=(exctype, value, traceback))


sys.excepthook = _exception_hook
# -- Setup style
# Load the CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("./style.css")

# -- Define some useful functions
PROB_CLASS_NAME = 'Prob_0'

@st.cache_data
def load_image_attention_pairs(img_dir: Path, id_globber: str = r"\w+\d+"):
    """Load and pair image files with their corresponding attention maps
    Both files should be under img_dir and share the same ID pattern
    Images have 'image' in filename, attention maps have 'pb_pred'"""

    # Glob all nifti files
    all_files = list(img_dir.rglob("*nii.gz"))

    # Separate into images and attention maps
    image_files = {re.search(id_globber, f.name).group(): f
                   for f in all_files if 'image' in f.name.lower()}
    attn_files = {re.search(id_globber, f.name).group(): f
                  for f in all_files if 'pb_pred' in f.name.lower()}

    # Find IDs that have both image and attention map
    intersection = list(set(image_files.keys()) & set(attn_files.keys()))
    intersection.sort()

    # Create pairs dictionary
    paired = {sid: (image_files[sid], attn_files[sid]) for sid in intersection}
    return paired


def save_batch_images(filtered_intersection, paired, seg_paired, output_dir,
                      window_range, attn_threshold, alpha, contour_alpha, contour_width,
                      head_settings, csv_data=None, slice_offset=0):
    """Save all images in the filtered list to the specified directory.

    Args:
        filtered_intersection (List[str]): List of IDs to process.
        paired (Dict[str, Tuple[Path, Path]]): Dictionary mapping IDs to (image_path, attention_path).
        seg_paired (Dict[str, Tuple[Path, Path]]): Dictionary mapping IDs to (image_path, segmentation_path).
        output_dir (Path): Path to save the images.
        window_range (Tuple[int, int]): Tuple of (lower, upper) for window level adjustments.
        attn_threshold (Tuple[int, int]): Tuple of (min, max) for attention thresholding.
        alpha (float): Opacity for overlaying attention maps.
        head_settings (Dict[str, Any]): Dictionary containing settings for attention head selection.

    Returns:
        List[Path]: List of successfully saved image paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    progress_bar = st.progress(0)

    # Final pred dict
    final_predictions = {}

    for idx, selected_pair in enumerate(filtered_intersection):
        try:
            # Get paths
            img_path, attn_path = paired[selected_pair]
            seg_path = seg_paired.get(selected_pair, None)[1] if len(seg_paired) else None

            # If there are more than one prob incoming, take first
            prob = csv_data.loc[selected_pair][PROB_CLASS_NAME] if csv_data is not None else None
            try:
                if len(prob) > 1:
                    logger.warning(f"There are more than one row for this ID: {prob}")
                    prob = prob[0]
            except:
                pass

            try:
                overlayed, _, _, final_prediction_text = create_display_image(
                    img_path=img_path,
                    attn_path=attn_path,
                    seg_path=seg_path,
                    window_range=window_range,
                    attn_threshold=attn_threshold,
                    alpha=alpha,
                    contour_alpha=contour_alpha,
                    contour_width=contour_width,
                    head_settings=head_settings,
                    case_id=selected_pair,
                    prob=prob,
                    slice_offset=slice_offset
                )
                final_predictions[selected_pair] = final_prediction_text

                # Save the image
                output_path = output_dir / f"{selected_pair}_overlay.png"
                cv2.imwrite(str(output_path), cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
                saved_paths.append(output_path)
            except Exception as e:
                st.error(f"Failed to save image for ID {selected_pair}: {e}")
                logger.exception(e)

                # Update progress
            progress_bar.progress((idx + 1) / len(filtered_intersection))

        except Exception as e:
            st.error(f"Failed to save image for ID {selected_pair}: {e}")
            logger.exception(e)
            continue

    return saved_paths, final_predictions


def build_configurations():
    # Get available states
    available_states = Configuration.list_available_states()
    
    # Load configuration
    if 'selected_state_file' not in st.session_state and available_states:
        st.session_state['selected_state_file'] = str(available_states[0])
    
    if st.session_state.get('selected_state_file') and Path(st.session_state['selected_state_file']).exists():
        conf_instance = Configuration.load_from_json(st.session_state['selected_state_file'])
    else:
        conf_instance = Configuration()

    var_mapping = Configuration.mapper
    
    # Initialize session state from config (only on first load)
    if 'config_initialized' not in st.session_state:
        for k, v in var_mapping.items():
            session_key = v.lower()
            setattr(st.session_state, session_key, getattr(conf_instance, v))
        
        # Initialize display settings
        for field in ['image_window_range', 'attn_threshold', 'attn_opacity', 'contour_alpha', 'contour_width', 'slice_offset']:
            st.session_state[field] = getattr(conf_instance, field)
        
        st.session_state['config_initialized'] = True

    with st.expander('Configurations', expanded=st.session_state.get('require_setup', True)):
        st.write("### Instructions")
        st.write("""
        - Insert the directories in absolute format or relative to the streamlit file
        - The ID globber is the regex that will be used to match the ID of the images and segmentations
        """)

        # State selector
        if available_states:
            state_options = [str(s) for s in available_states]
            selected_state = st.selectbox(
                "Load State",
                options=state_options,
                index=state_options.index(st.session_state.get('selected_state_file', state_options[0])) if st.session_state.get('selected_state_file') in state_options else 0,
                help="Select a saved state to load"
            )
            if selected_state != st.session_state.get('selected_state_file'):
                logger.info(f"Loading state from {selected_state}")
                st.session_state['selected_state_file'] = selected_state
                
                # Reload configuration and update session state
                new_conf = Configuration.load_from_json(selected_state)
                for k, v in var_mapping.items():
                    session_key = v.lower()
                    setattr(st.session_state, session_key, getattr(new_conf, v))
                
                # Update display settings
                for field in ['image_window_range', 'attn_threshold', 'attn_opacity', 'contour_alpha', 'contour_width', 'slice_offset']:
                    st.session_state[field] = getattr(new_conf, field)
                
                st.rerun()

        with st.form('config_form'):
            # ===== Form start =====
            for k, v in var_mapping.items():
                var_type = type(getattr(conf_instance, v))

                if var_type == str:
                    if 'DIR' in v:
                        setattr(st.session_state, v.lower(), Path(st.text_input(
                            k,
                            value=str(st.session_state.get(v.lower(), getattr(conf_instance, v)))
                        )))
                    else:
                        setattr(st.session_state, v.lower(), st.text_input(
                            k,
                            value=st.session_state.get(v.lower(), getattr(conf_instance, v))
                        ))


            # Buttons
            col1, col2, _ = st.columns([1, 1, 3])
            with col1:
                st.form_submit_button("Apply Configurations")

            with col2:
                st.checkbox("Plot with segmentation", key="USE_SEGMENT")
                # ====== Form end =======
                
        if st.button("Save States", key="save_state_btn"):
            # Update config from session state
            for field_name, field in Configuration.model_fields.items():
                session_key = field_name.lower()
                if session_key in st.session_state:
                    value = st.session_state[session_key]
                    if isinstance(value, Path):
                        value = str(value)
                    setattr(conf_instance, field_name, value)

            # Save with image-based filename
            image_path = st.session_state.get('image_dir', '.')
            saved_path = Configuration.save_state(conf_instance, str(image_path))
            st.session_state['selected_state_file'] = str(saved_path)
            st.success(f"State saved to {saved_path.name}")
            st.rerun()


if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    build_configurations()
else:
    build_configurations()

# Target ID list
with st.expander("Specify ID"):
    target_ids = st.text_input("CSV string", value="")
    if len(target_ids):
        target_ids = list(set(target_ids.split(',')))

# Load the pairs
image_dir = st.session_state.image_dir
id_globber = st.session_state.id_globber

if image_dir.is_dir():
    # Load image files
    all_files = list(image_dir.rglob("*nii.gz"))
    logger.info(f"Found {len(all_files)} nifti files in image_dir")
    image_files = {re.search(id_globber, f.name).group(): f
                   for f in all_files if 'image' in f.stem.lower()}
    logger.info(f"Found {len(image_files)} unique keys: {'.'.join(image_files.keys())}")

    # Load attention maps if directory is configured
    attn_files = {}
    if st.session_state.attn_dir and Path(st.session_state.attn_dir).is_dir():
        attn_files = {re.search(id_globber, f.name).group(): f
                      for f in Path(st.session_state.attn_dir).rglob("*nii.gz")
                      if re.search(r'(?i)(heatmap|pb_map|pb_pred)', f.name.lower())}
        st.success(f"Successfully loaded {len(attn_files)} attention maps")

        # Check for images without attention maps
        images_without_attn = set(image_files.keys()) - set(attn_files.keys())
        if images_without_attn:
            st.warning(
                f"Found {len(images_without_attn)} images without attention maps: {','.join(sorted(images_without_attn))}")

    # Create pairs dictionary
    if attn_files:
        # If attention maps are available, create pairs with both
        intersection = list(set(image_files.keys()) & set(attn_files.keys()))
        paired = {sid: (image_files[sid], attn_files[sid]) for sid in intersection}
    else:
        # If no attention maps, just use images
        paired = {sid: (image_files[sid], None) for sid in image_files.keys()}

    intersection = list(paired.keys())
    intersection.sort()

    # Filter by target_ids if specified
    if len(target_ids):
        intersection = set(intersection) & set(target_ids)
        if missing := set(target_ids) - set(intersection):
            st.warning(f"IDs specified but the following are missing: {','.join(missing)}")
        intersection = list(intersection)
    st.session_state.require_setup = False
else:
    st.error(f"Directory `{str(image_dir)}` does not exist!")
    st.stop()

# Load segmentation pairs if segmentation directory is configured
seg_dir = st.session_state.segmentation_dir
seg_paired = {}
if seg_dir != "" and Path(seg_dir).is_dir() and st.session_state.USE_SEGMENT:
    try:
        seg_files = {re.search(id_globber, f.name).group(): f
                     for f in Path(seg_dir).rglob("*nii.gz")}

        # Check for images without segmentations
        images_without_seg = set(image_files.keys()) - set(seg_files.keys())
        if images_without_seg:
            st.warning(
                f"Found {len(images_without_seg)} images without segmentation masks: {','.join(sorted(images_without_seg))}")

        seg_paired = {sid: (image_files[sid], seg_files[sid])
                      for sid in set(image_files.keys()) & set(seg_files.keys())}
        st.toast(f"Successfully loaded {len(seg_paired)} segmentation masks", duration='short')
    except Exception as e:
        st.warning(f"Failed to load segmentation masks: {e}")
        logger.exception(e)
else:
    if not Path(seg_dir).is_dir() and st.session_state.USE_SEGMENT:
        st.warning(f"Segmentation directory `{str(seg_dir)}` does not exist!")
    else:
        st.info("Segmentation is not loaded because its not specified")

# Load the csv file
csv_dir = st.session_state.csv_dir
if csv_dir.is_file():
    csv_data = pd.read_csv(csv_dir, index_col=0)
    csv_data.rename({'Prob_Class_0':PROB_CLASS_NAME}, axis=1, inplace=True)

    # If the csv is from simple inference, it will have a different format, handling it here
    # note that we require "Truth_0" column to be present. Otherwise, it's not meaningful to
    # perform these transforms.
    logger.debug(f"{csv_data.columns}")
    if all(col in csv_data.columns for col in ['OverallPrediction', 'Truth_0']):
        logger.info(f"Detected results from simple_inference, transforming columns.")


        def _sigmoid(val):
            return 1 / (1 + np.exp(-val))

        logger.info("Adding Prob_0")
        csv_data[PROB_CLASS_NAME] = csv_data['OverallPrediction'].astype(float).apply(_sigmoid)

        logger.info("Adding Decision_0")
        csv_data['Decision_0'] = csv_data[PROB_CLASS_NAME] > 0.5

    if 'Decision_0' in csv_data.columns and 'Truth_0' in csv_data.columns:
        logger.info("Calculating confusion matrix categories")
        # Do basic checking
        if set(csv_data['Decision_0'].unique().tolist()) - {0, 1, True, False}:
            logger.warning("The 'Decision_0' column contains values other than 0/1/True/False. This may cause issues with filtering and statistics.")
            st.warning(f"The 'Decision_0' column contains values other than 0/1/True/False. This may cause issues with filtering and statistics."
                       f"{csv_data['Decision_0'].unique()}")
        if set(csv_data['Truth_0'].unique().tolist()) - {0, 1, True, False}:
            logger.warning("The 'Truth_0' column contains values other than 0/1/True/False. This may cause issues with filtering and statistics.")
            st.warning(f"The 'Truth_0' column contains values other than 0/1/True/False. This may cause issues with filtering and statistics."
                       f"{csv_data['Truth_0'].unique()}")

        csv_data['Confusion_Matrix'] = 'Unknown'
        csv_data['Decision_0'] = csv_data['Decision_0'].astype(int)
        csv_data['Truth_0'] = csv_data['Truth_0'].astype(int)
        csv_data.loc[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 1), 'Confusion_Matrix'] = 'TP'
        csv_data.loc[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 0), 'Confusion_Matrix'] = 'TN'
        csv_data.loc[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 0), 'Confusion_Matrix'] = 'FP'
        csv_data.loc[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 1), 'Confusion_Matrix'] = 'FN'

    st.dataframe(csv_data, key="display_data")

else:
    st.error(f"File `{str(csv_dir)}` does not exist!")
    st.stop()

# -- Streamlit app
st.title("Image and Attention Map Viewer")

# Initialize session state
if 'selection_index' not in st.session_state:
    st.session_state.selection_index = 0
if 'filtered_intersection' not in st.session_state:
    st.session_state.filtered_intersection = intersection

# Use the filtered list if the filtered_intersection is not empty
filtered_intersection = st.session_state.filtered_intersection
filtered_intersection.sort()
selected_index = st.selectbox("Slice image pair", range(len(filtered_intersection)),
                              format_func=lambda x: filtered_intersection[x],
                              index=min(st.session_state.selection_index,
                                        len(filtered_intersection) - 1) if filtered_intersection else 0)

if not selected_index == st.session_state.selection_index:
    st.session_state.selection_index = selected_index
    st.rerun()

# Use try-except to catch user input that doesn't exist
try:
    selected_pair = str(filtered_intersection[selected_index])
    # st.write(paired[selected_pair])
except:
    st.write("The selected ID is not in the record.")
    st.stop()

# If selected pair found
if selected_pair:
    with st.container(height=700):
        image_slot = st.empty()

    st.write("### Prediction Results")
    st.write("The prediction results are shown below. The first column is the ID of the image, "
             "and the second column is the raw prediction results (pre-sigmoid), the third column is the "
             "prediction results after sigmoid. The fourth column is the ground-truth label.")
    dataframe_slot = st.empty()
    with st.expander("📊 Full Prediction Results"):
        # Add filtering radio buttons for TP/TN/FP/FN
        filter_option = st.radio(
            "Filter prediction results:",
            ["All", "True Positives (TP)", "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)"],
            key="filter_option"
        )

        # Check if filter option has changed
        if 'previous_filter_option' not in st.session_state:
            st.session_state.previous_filter_option = filter_option

        filter_changed = st.session_state.previous_filter_option != filter_option
        if filter_changed:
            # Reset selection index when filter changes
            st.session_state.selection_index = 0
            st.session_state.previous_filter_option = filter_option

        # Create a filtered dataframe based on selection
        filtered_csv_data = csv_data.copy()
        filtered_ids = intersection.copy()

        # Calculate statistics for all categories
        if 'Decision_0' in csv_data.columns and 'Truth_0' in csv_data.columns:
            tp_count = len(csv_data[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 1)])
            tn_count = len(csv_data[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 0)])
            fp_count = len(csv_data[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 0)])
            fn_count = len(csv_data[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 1)])

            # Calculate metrics
            total = tp_count + tn_count + fp_count + fn_count
            accuracy = (tp_count + tn_count) / total if total > 0 else 0
            sensitivity = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            specificity = tn_count / (tn_count + fp_count) if (tn_count + fp_count) > 0 else 0

            # Display metrics
            st.write(f"### Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Positives (TP)", tp_count)
                st.metric("False Positives (FP)", fp_count)
            with col2:
                st.metric("True Negatives (TN)", tn_count)
                st.metric("False Negatives (FN)", fn_count)
            with col3:
                st.metric("Accuracy", f"{accuracy:.2%}")
                st.metric("Sensitivity", f"{sensitivity:.2%}")
                st.metric("Specificity", f"{specificity:.2%}")

        # Apply filtering based on selection
        if filter_option != "All":
            # Let's determine TP/TN/FP/FN based on prediction and ground truth
            # Assuming the format is: raw prediction, probability, decision (0/1), ground truth (0/1)
            if 'Decision_0' in csv_data.columns and 'Truth_0' in csv_data.columns:
                if filter_option == "True Positives (TP)":
                    # Both prediction and truth are positive (1)
                    filtered_csv_data = csv_data[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 1)]
                elif filter_option == "True Negatives (TN)":
                    # Both prediction and truth are negative (0)
                    filtered_csv_data = csv_data[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 0)]
                elif filter_option == "False Positives (FP)":
                    # Prediction is positive (1) but truth is negative (0)
                    filtered_csv_data = csv_data[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 0)]
                elif filter_option == "False Negatives (FN)":
                    # Prediction is negative (0) but truth is positive (1)
                    filtered_csv_data = csv_data[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 1)]

                # Update the intersection list to only include IDs from the filtered dataframe
                filtered_ids = [id for id in intersection if id in filtered_csv_data.index]

                # Update the selection options if we're filtering
                if filtered_ids:
                    # Store filtered IDs in session state for access by selectbox
                    st.session_state.filtered_intersection = filtered_ids
                    if filter_changed:
                        st.rerun()  # Rerun to update the UI with the new filter
                else:
                    st.warning(f"No cases found for the selected filter: {filter_option}")
                    st.session_state.filtered_intersection = intersection
            else:
                st.warning("CSV data does not contain the required columns (Decision_0, Truth_0) for filtering")
                st.session_state.filtered_intersection = intersection
        else:
            # If "All" is selected, use the original intersection
            st.session_state.filtered_intersection = intersection
            if filter_changed:
                st.rerun()  # Rerun to update the UI when switching back to "All"

        # Show current filter status
        st.write(f"### Showing {len(filtered_csv_data)} cases for filter: {filter_option}")

        # Display the dataframe (either filtered or all)
        st.dataframe(filtered_csv_data)

    with st.expander("⚙️ Display settings"):
        st.write("### Image window-level and attention map settings")
        col1, spacer, col2 = st.columns([1, 0.1, 1])
        with col1:
            # Slider to control the window range of the image
            lower, upper = st.slider(
                'Image window range',
                min_value=0,
                max_value=99,
                value=st.session_state.get('image_window_range', Configuration.DEFAULT_WINDOW_RANGE)
            )
            st.session_state['image_window_range'] = (lower, upper)

            # Slider to control the threshold for the attention map
            attn_min, attn_max = st.slider(
                'Attention map threshold',
                min_value=Configuration.ATTN_MIN_VALUE,
                max_value=Configuration.ATTN_MAX_VALUE,
                value=st.session_state.get('attn_threshold', Configuration.DEFAULT_ATTN_THRESHOLD)
            )
            st.session_state['attn_threshold'] = (attn_min, attn_max)

            # Slider to control the opacity of the attention map
            alpha = st.slider(
                'Attention map opacity',
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('attn_opacity', Configuration.DEFAULT_OPACITY)
            )
            st.session_state['attn_opacity'] = alpha

            contour_alpha = st.slider(
                'Contour alpha',
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('contour_alpha', Configuration.DEFAULT_CONTOUR_ALPHA)
            )
            st.session_state['contour_alpha'] = contour_alpha

            contour_width = st.number_input('Contour Width', min_value=1, max_value=5, step=1)

            slice_offset = st.number_input(
                'Slice Offset',
                min_value=0,
                step=1,
                value=st.session_state.get('slice_offset', Configuration.DEFAULT_SLICE_OFFSET),
                help="Number of slices to skip from the start of the volume before building the grid display"
            )
            st.session_state['slice_offset'] = slice_offset

        with col2:
            st.write("### Attention Head Selection")
            # Add controls for attention head selection
            head_col1, head_col2 = st.columns([1, 1])
            with head_col1:
                use_max = st.checkbox("Use Max", value=False)
            with head_col1:
                use_avg = st.checkbox("Use Avg", value=False)
            with head_col1:
                head_idx = st.number_input(
                    'Select head',
                    min_value=0,
                    max_value=st.session_state.get('num_heads', 20),
                    value=0,
                    step=1,
                    disabled=(use_max or use_avg)
                )

    with st.spinner("Loading...") as spinner:
        img_path, attn_path = paired[selected_pair]
        seg_path = seg_paired.get(selected_pair, None) if len(seg_paired) else None
        if seg_path is not None:
            seg_path = seg_path[1]
            logger.debug(f"Loading segmentation: {seg_path}")

        # If there are more than one prob incoming, take first
        prob = csv_data.loc[selected_pair][PROB_CLASS_NAME]
        try:
            if len(prob) > 1:
                logger.warning(f"There are more than one row for this ID: {prob}")
                prob = prob[0]
        except:
            pass

        try:
            # Create the image using the new function
            IMAGE_DISPLAY_ERROR = False
            overlayed, attn_map_target, img_sitk, _ = create_display_image(
                img_path=img_path,
                attn_path=attn_path,
                seg_path=seg_path,
                window_range=(lower, upper),
                attn_threshold=(attn_min, attn_max),
                alpha=alpha,
                contour_alpha=contour_alpha,
                contour_width=contour_width,
                head_settings={'use_max': use_max, 'use_avg': use_avg, 'head_idx': head_idx},
                case_id=selected_pair,
                prob=prob,
                slice_offset=st.session_state.get('slice_offset', 0)
            )

            # Show the image
            image_slot.image(overlayed, width='stretch')
        except Exception as e:
            st.error(f"Cannot process this case. Original error: {e}")
            logger.exception(e)
            IMAGE_DISPLAY_ERROR=True

    if IMAGE_DISPLAY_ERROR:
        st.stop() # need to put stop here to prevent spinner keeps spinning. 

    # Load the result data and display them
    if selected_pair in filtered_csv_data.index:
        logger.debug(f"Selected pair: {selected_pair}")
        dataframe_slot.dataframe(filtered_csv_data.loc[selected_pair].to_frame().T)
    else:
        st.warning(f"Selected ID {selected_pair} is not found in the filtered dataset.")
        dataframe_slot.dataframe(pd.DataFrame())

    # Add histogram plot of attention map with threshold lines only if attention map is available
    if attn_map_target is not None:
        with st.expander("📊 Plots"):
            attn_values = attn_map_target.flatten()

            # Filter out extreme values for better visualization
            lower_percentile = Configuration.HIST_LOWER_PERCENTILE
            upper_percentile = Configuration.HIST_UPPER_PERCENTILE
            p1 = np.percentile(attn_values, lower_percentile)
            p2 = np.percentile(attn_values, upper_percentile)
            filtered_values = attn_values[(attn_values >= p1) & (attn_values <= p2)]

            # Create histogram using plotly
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=attn_values.flatten(),
                xbins=dict(start=max(2, attn_min - 40), end=min(255, attn_max + 40), size=1),
                histnorm='probability density',
                name='Distribution'
            ))

            # Add vertical lines for threshold values if applicable
            if attn_min < attn_max:
                # Convert threshold from 0-255 scale to actual data range
                actual_min = attn_min
                actual_max = attn_max

                fig.add_vline(
                    x=actual_min,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Min ({attn_min})",
                    annotation_position="top right"
                )
                fig.add_vline(
                    x=actual_max,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Max ({attn_max})",
                    annotation_position="top left"
                )

            fig.update_layout(
                title=f'Attention Map Distribution ({int(lower_percentile)}-{int(upper_percentile)}th percentile)',
                xaxis_title='Attention Value',
                yaxis_title='Density',
                xaxis_range=[max(2, attn_min - 40), min(255, attn_max + 40)],
                showlegend=False
            )

            st.plotly_chart(fig)

    # Previous button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button('⬅️ Previous', width='stretch'):
            current_index = selected_index
            previous_index = (current_index - 1) % len(filtered_intersection)
            st.session_state.selection_index = previous_index
            # No need to rerun becasue the frames update automatically

    # Next button
    with col2:
        if st.button('Next ➡️', width='stretch'):
            current_index = selected_index
            next_index = (current_index + 1) % len(filtered_intersection) if filtered_intersection else 0
            st.session_state.selection_index = next_index
            # No need to rerun becasue the frames update automatically

    with col3:
        with st.expander("💾 Save Images"):
            output_dir = st.session_state['output_dir'] 
            output_dir = Path(output_dir)

            save_col1, save_col2 = st.columns([1, 1])
            with save_col1:
                if st.button('Save Current Image', width='stretch'):
                    st.write("Saving...")
                    try:
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Get the image and attention map paths
                        img_path, attn_path = paired[selected_pair]

                        # Save the images to temp directory
                        image_name = Path(img_path).stem
                        output_path = output_dir / f"{image_name}_processed.png"

                        # Save the image
                        cv2.imwrite(str(output_path), cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))

                        st.success(f"Images saved successfully to:\n{output_path}")
                    except Exception as e:
                        st.error(f"Failed to save images: {e}")

            with save_col2:
                if st.button('Save All Filtered Images', width='stretch'):
                    # Create a directory selector
                    if output_dir:
                        st.write("Saving all filtered images...")
                        try:
                            saved_paths, final_predictions = save_batch_images(
                                filtered_intersection=filtered_intersection,
                                paired=paired,
                                seg_paired=seg_paired,
                                output_dir=output_dir,
                                window_range=(lower, upper),
                                attn_threshold=(attn_min, attn_max),
                                alpha=alpha,
                                contour_alpha=contour_alpha,
                                contour_width=contour_width,
                                head_settings={'use_max': use_max, 'use_avg': use_avg, 'head_idx': head_idx},
                                csv_data=filtered_csv_data,
                                slice_offset=st.session_state.get('slice_offset', 0)
                            )

                            # Add the final outcome decision into the dataframe
                            _df = filtered_csv_data.copy()
                            _s = pd.Series(final_predictions, name='Final Predictions')
                            _df = _df.join(_s)

                            # Save the dataframe
                            csv_output_path = output_dir / "filtered_results.csv"
                            _df.to_csv(csv_output_path)

                            st.success(f"Successfully saved {len(saved_paths)} images and results CSV to {output_dir}")
                        except Exception as e:
                            st.error(f"Failed to save batch images: {e}")
                    else:
                        st.error(f"Specify the output directory")
                        st.stop()
                st.write("Note that this will also save the filtered reuslts to 'filtered_results.csv'")

