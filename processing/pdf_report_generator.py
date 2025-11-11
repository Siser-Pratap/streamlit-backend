# Import required libraries and modules
import os
import io
import base64
import datetime
import glob
import tempfile
import shutil
from typing import List, Tuple, Dict, Any

from utils.logger import logger

import pandas as pd
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
MATPLOTLIB_AVAILABLE = True
from fastapi import UploadFile



# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing, Rect, Line
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics import renderPDF
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Plotly imports for 3D graphs
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from dotenv import load_dotenv
from processing.cloudinary import upload_to_cloudinary

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chart generation configuration
CHART_STYLE = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'whitegrid',
    'palette': 'Set2',
    'font_scale': 1.2
}

# Set up matplotlib/seaborn styling
if MATPLOTLIB_AVAILABLE:
    plt.style.use('default')
    sns.set_style(CHART_STYLE['style'])
    sns.set_palette(CHART_STYLE['palette'])
    sns.set_context("notebook", font_scale=CHART_STYLE['font_scale'])


def create_temp_chart_directory() -> str:
    """
    Create a temporary directory for storing chart images
    """
    temp_dir = tempfile.mkdtemp(prefix="rantau_charts_")
    logger.info(f"Created temporary chart directory: {temp_dir}")
    return temp_dir


def cleanup_temp_directory(temp_dir: str):
    """
    Clean up temporary directory and all its contents
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary directory {temp_dir}: {e}")


def save_chart_as_png(fig: Figure, filename: str, chart_dir: str = None, dpi: int = 300) -> str:
    """
    Save matplotlib figure as PNG and upload to Cloudinary
    Returns Cloudinary URL instead of local file path
    """
    # Clean filename (remove invalid characters)
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_filename = safe_filename.replace(' ', '_')
    
    # Save figure to BytesIO buffer first (before closing figure)
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        plt.close(fig)  # Close figure to free memory
    except Exception as save_error:
        logger.error(f"Error saving figure to buffer {filename}: {save_error}")
        plt.close(fig)
        return None
    
    # Try uploading to Cloudinary first
    try:
        cloudinary_url = upload_to_cloudinary(image_bytes, folder="charts")
        logger.info(f"Uploaded chart to Cloudinary: {cloudinary_url}")
        return cloudinary_url
    except Exception as e:
        logger.warning(f"Cloudinary upload failed for {filename}: {e}")
        # Fallback to local save if Cloudinary fails
        try:
            if chart_dir is None:
                chart_dir = os.path.join(os.getcwd(), "assets")
            os.makedirs(chart_dir, exist_ok=True)
            filepath = os.path.join(chart_dir, f"{safe_filename}.png")
            
            # Write buffer to file
            buffer.seek(0)  # Reset buffer position
            with open(filepath, 'wb') as f:
                f.write(buffer.getvalue())
            
            logger.warning(f"Cloudinary upload failed, saved locally: {filepath}")
            return filepath
        except Exception as fallback_error:
            logger.error(f"Fallback local save also failed: {fallback_error}")
            return None


def generate_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, 
                      chart_dir: str, brand_name: str = None) -> str:
    """
    Generate a bar chart using seaborn and save as PNG
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, cannot generate bar chart")
        return None
    
    try:
        fig, ax = plt.subplots(figsize=CHART_STYLE['figure_size'])
        
        # Create bar plot
        sns.barplot(data=data, x=x_col, y=y_col, ax=ax, palette=CHART_STYLE['palette'])
        
        # Customize the chart
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
        
        # Rotate x-axis labels if needed
        if len(data[x_col].unique()) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Add brand logo if available
        if brand_name:
            add_brand_watermark(fig, brand_name)
        
        # Save chart
        filename = f"bar_chart_{x_col}_{y_col}".replace(' ', '_').lower()
        return save_chart_as_png(fig, filename, chart_dir)
        
    except Exception as e:
        logger.error(f"Error generating bar chart: {e}")
        return None


def generate_line_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, 
                       chart_dir: str, brand_name: str = None) -> str:
    """
    Generate a line chart using seaborn and save as PNG
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, cannot generate line chart")
        return None
    
    try:
        fig, ax = plt.subplots(figsize=CHART_STYLE['figure_size'])
        
        # Create line plot
        sns.lineplot(data=data, x=x_col, y=y_col, ax=ax, marker='o', linewidth=2.5)
        
        # Customize the chart
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
        
        # Format x-axis for dates if applicable
        if pd.api.types.is_datetime64_any_dtype(data[x_col]):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Add brand logo if available
        if brand_name:
            add_brand_watermark(fig, brand_name)
        
        # Save chart
        filename = f"line_chart_{x_col}_{y_col}".replace(' ', '_').lower()
        return save_chart_as_png(fig, filename, chart_dir)
        
    except Exception as e:
        logger.error(f"Error generating line chart: {e}")
        return None


def generate_pie_chart(data: pd.DataFrame, value_col: str, label_col: str, title: str,
                      chart_dir: str, brand_name: str = None) -> str:
    """
    Generate a clear, high-resolution pie chart using matplotlib and save as PNG
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, cannot generate pie chart")
        return None
    
    try:
        fig, ax = plt.subplots(figsize=CHART_STYLE.get('figure_size', (6, 6)), dpi=150)
        
        # Create pie chart with improved spacing
        colors = sns.color_palette(CHART_STYLE.get('palette', 'pastel'), len(data))
        wedges, texts, autotexts = ax.pie(
            data[value_col],
            labels=None,  # Hide labels inside slices
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            pctdistance=0.8
        )
        
        # Add legend instead of overlapping labels
        ax.legend(
            wedges,
            data[label_col],
            title="",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Title styling
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Improve readability of percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Equal aspect ratio for a perfect circle
        ax.axis('equal')
        
        # Add watermark if brand is provided
        if brand_name:
            add_brand_watermark(fig, brand_name)
        
        # Tight layout to avoid cutoff
        plt.tight_layout()
        
        # Save at high resolution
        filename = f"pie_chart_{value_col}_{label_col}".replace(' ', '_').lower()
        filepath = save_chart_as_png(fig, filename, chart_dir, dpi=200)
        
        plt.close(fig)
        return filepath
    
    except Exception as e:
        logger.error(f"Error generating pie chart: {e}")
        return None



def generate_scatter_plot(data: pd.DataFrame, x_col: str, y_col: str, title: str, 
                         chart_dir: str, brand_name: str = None, hue_col: str = None) -> str:
    """
    Generate a scatter plot using seaborn and save as PNG
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, cannot generate scatter plot")
        return None
    
    try:
        fig, ax = plt.subplots(figsize=CHART_STYLE['figure_size'])
        
        # Create scatter plot
        if hue_col and hue_col in data.columns:
            sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax, s=100)
        else:
            sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax, s=100)
        
        # Customize the chart
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Add brand logo if available
        if brand_name:
            add_brand_watermark(fig, brand_name)
        
        # Save chart
        filename = f"scatter_plot_{x_col}_{y_col}".replace(' ', '_').lower()
        return save_chart_as_png(fig, filename, chart_dir)
        
    except Exception as e:
        logger.error(f"Error generating scatter plot: {e}")
        return None


def generate_heatmap(data: pd.DataFrame, title: str, chart_dir: str, 
                    brand_name: str = None, annot: bool = True) -> str:
    """
    Generate a heatmap using seaborn and save as PNG
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, cannot generate heatmap")
        return None
    
    try:
        fig, ax = plt.subplots(figsize=CHART_STYLE['figure_size'])
        
        # Create heatmap
        sns.heatmap(data, annot=annot, cmap='YlOrRd', ax=ax, cbar_kws={'shrink': 0.8})
        
        # Customize the chart
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add brand logo if available
        if brand_name:
            add_brand_watermark(fig, brand_name)
        
        # Save chart
        filename = f"heatmap_{title}".replace(' ', '_').lower()
        return save_chart_as_png(fig, filename, chart_dir)
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return None


def add_brand_watermark(fig: Figure, brand_name: str):
    """
    Add brand watermark to chart
    """
    try:
        # Add text watermark
        fig.text(0.02, 0.02, f"© {brand_name}", fontsize=8, alpha=0.5, 
                color='gray', ha='left', va='bottom')
    except Exception as e:
        logger.warning(f"Could not add brand watermark: {e}")


def generate_charts_from_data(summary: List[Dict], 
                             brand_name: str = None, 
                             main_df: pd.DataFrame = None) -> List[Tuple[str, str]]:
    """
    Generate various charts from the data and return list of (title, filepath) tuples
    Charts are saved in assets directory
    Enhanced with comprehensive chart generation based on data analysis
    
    Args:
        summary: List of dicts from summary sheet (for backward compatibility)
        brand_name: Brand name for watermarking
        main_df: Main DataFrame from "Complete sheet" with actual data points
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, cannot generate charts")
        return []
    
    charts = []
    # Chart directory is no longer needed for Cloudinary, but kept for backward compatibility
    chart_dir = None
    
    try:
        # Use main_df if provided, otherwise try to convert summary to DataFrame
        if main_df is not None and not main_df.empty:
            df = main_df.copy()
            logger.info(f"Using main DataFrame with {len(df)} rows for chart generation")
        else:
            # Fallback: convert summary to DataFrame
            df = pd.DataFrame(summary) if summary else pd.DataFrame()
            logger.info(f"Using summary data converted to DataFrame with {len(df)} rows")
        
        # Normalize column names (handle case variations)
        if not df.empty:
            df.columns = [col.strip() for col in df.columns]
            # Create lowercase mapping for easier column access
            col_lower_map = {col.lower(): col for col in df.columns}
            
            # Generate charts based on available data from gentari.py output structure
        if not df.empty:
            
            
            # 1. Sentiment Distribution (Bar Chart) - from "Sentiment" column
            sentiment_col = col_lower_map.get('sentiment')
            if sentiment_col:
                sentiment_counts = df[sentiment_col].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                if not sentiment_counts.empty:
                    chart_path = generate_bar_chart(
                        sentiment_counts, 'Sentiment', 'Count', 
                        'Sentiment Distribution', chart_dir, brand_name
                    )
                    if chart_path:
                        charts.append(("Sentiment Distribution", chart_path))
            
            # 2. Type Distribution (Bar Chart) - from "Type" column
            type_col = col_lower_map.get('type')
            if type_col:
                type_counts = df[type_col].value_counts().reset_index()
                type_counts.columns = ['Type', 'Count']
                if not type_counts.empty:
                    chart_path = generate_bar_chart(
                        type_counts, 'Type', 'Count', 
                        'Media Type Distribution', chart_dir, brand_name
                    )
                    if chart_path:
                        charts.append(("Media Type Distribution", chart_path))
            
            # 3. Category Distribution (Bar Chart) - from "Category" column
            category_col = col_lower_map.get('category')
            if category_col:
                category_counts = df[category_col].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']
                if not category_counts.empty:
                    chart_path = generate_bar_chart(
                        category_counts, 'Category', 'Count', 
                        'Category Distribution', chart_dir, brand_name
                    )
                    if chart_path:
                        charts.append(("Category Distribution", chart_path))
            
            # 4. Media Impression Over Time (Line Chart) - from "Date" and "Media Impression" columns
            date_col = col_lower_map.get('date')
            media_impression_col = col_lower_map.get('media impression')
            if date_col and media_impression_col:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    time_series = df.groupby(date_col)[media_impression_col].sum().reset_index()
                    time_series.columns = ['Date', 'Total Media Impression']
                    time_series = time_series.dropna()
                    if not time_series.empty:
                        chart_path = generate_line_chart(
                            time_series, 'Date', 'Total Media Impression', 
                            'Media Impression Trends Over Time', chart_dir, brand_name
                        )
                        if chart_path:
                            charts.append(("Media Impression Trends Over Time", chart_path))
                except Exception as e:
                    logger.warning(f"Could not create time series chart: {e}")
            
            # 5. Social Impressions Over Time (Line Chart)
            social_impression_col = col_lower_map.get('social impressions')
            if date_col and social_impression_col:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    social_series = df.groupby(date_col)[social_impression_col].sum().reset_index()
                    social_series.columns = ['Date', 'Total Social Impressions']
                    social_series = social_series.dropna()
                    if not social_series.empty:
                        chart_path = generate_line_chart(
                            social_series, 'Date', 'Total Social Impressions', 
                            'Social Impressions Trends Over Time', chart_dir, brand_name
                        )
                        if chart_path:
                            charts.append(("Social Impressions Trends Over Time", chart_path))
                except Exception as e:
                    logger.warning(f"Could not create social impressions chart: {e}")
            
            # 6. Media Reach by Category (Bar Chart)
            media_reach_col = col_lower_map.get('media reach')
            if category_col and media_reach_col:
                try:
                    reach_by_category = df.groupby(category_col)[media_reach_col].sum().reset_index()
                    reach_by_category.columns = ['Category', 'Total Media Reach']
                    reach_by_category = reach_by_category.sort_values('Total Media Reach', ascending=False)
                    if not reach_by_category.empty:
                        chart_path = generate_bar_chart(
                            reach_by_category, 'Category', 'Total Media Reach', 
                            'Media Reach by Category', chart_dir, brand_name
                        )
                        if chart_path:
                            charts.append(("Media Reach by Category", chart_path))
                except Exception as e:
                    logger.warning(f"Could not create reach by category chart: {e}")
            
            # 7. Sentiment Pie Chart
            if sentiment_col:
                sentiment_counts = df[sentiment_col].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                if not sentiment_counts.empty and len(sentiment_counts) > 1:
                    chart_path = generate_pie_chart(
                        sentiment_counts, 'Count', 'Sentiment', 
                        'Sentiment Share', chart_dir, brand_name
                    )
                    if chart_path:
                        charts.append(("Sentiment Share", chart_path))
            
            # 8. Type Pie Chart
            if type_col:
                type_counts = df[type_col].value_counts().reset_index()
                type_counts.columns = ['Type', 'Count']
                if not type_counts.empty and len(type_counts) > 1:
                    chart_path = generate_pie_chart(
                        type_counts, 'Count', 'Type', 
                        'Media Type Share', chart_dir, brand_name
                    )
                    if chart_path:
                        charts.append(("Media Type Share", chart_path))
            
            # 9. Correlation between Media Reach and Impressions (Scatter Plot)
            if media_reach_col and media_impression_col:
                try:
                    scatter_data = df[[media_reach_col, media_impression_col]].dropna()
                    if not scatter_data.empty:
                        scatter_data.columns = ['Media Reach', 'Media Impression']
                        chart_path = generate_scatter_plot(
                            scatter_data, 'Media Reach', 'Media Impression', 
                            'Media Reach vs Media Impression', chart_dir, brand_name
                        )
                        if chart_path:
                            charts.append(("Media Reach vs Media Impression", chart_path))
                except Exception as e:
                    logger.warning(f"Could not create scatter plot: {e}")
        
        logger.info(f"Generated {len(charts)} charts successfully in assets directory")
        
    except Exception as e:
        logger.error(f"Error generating charts: {e}")
        return []
    
    return charts


def extract_excel_data(excel_bytes: bytes) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Extract data from Excel file and return both summary data and main DataFrame for chart generation
    Returns: (summary_list, main_dataframe)
    """
    try:
        # Read all sheets at once for better efficiency
        all_sheets = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
        
        # Extract Summary data (for insights generation)
        summary = []
        if 'summary' in all_sheets:
            summary = all_sheets['summary'].to_dict('records')
        else:
            logger.warning("summary tab not found in Excel file")
        
        # Extract main data sheet for chart generation
        # Try "Complete sheet" first (from gentari.py output)
        main_df = pd.DataFrame()
        if 'Complete sheet' in all_sheets:
            main_df = all_sheets['Complete sheet'].copy()
            logger.info(f"Found 'Complete sheet' with {len(main_df)} rows and columns: {list(main_df.columns)}")
        else:
            # Try other possible sheet names
            possible_sheet_names = ['data', 'Data', 'main', 'Main', 'Sheet1', 'Sheet 1']
            for sheet_name in possible_sheet_names:
                if sheet_name in all_sheets:
                    main_df = all_sheets[sheet_name].copy()
                    logger.info(f"Found '{sheet_name}' with {len(main_df)} rows")
                    break
        
        # If no main sheet found, try to use the first non-summary, non-data_chart sheet
        if main_df.empty:
            for sheet_name, sheet_df in all_sheets.items():
                if sheet_name.lower() not in ['summary', 'data_chart', 'data chart']:
                    # Check if it has the expected columns from gentari.py output
                    expected_cols = ['Date', 'Category', 'Type', 'Sentiment', 'Media Reach', 'Media Impression', 'Social Impressions']
                    if any(col in sheet_df.columns for col in expected_cols):
                        main_df = sheet_df.copy()
                        logger.info(f"Using '{sheet_name}' as main data sheet with {len(main_df)} rows")
                        break
        
        # Normalize column names to lowercase for easier access
        if not main_df.empty:
            main_df.columns = [col.strip() for col in main_df.columns]
            logger.info(f"Main DataFrame columns: {list(main_df.columns)}")
        
        return summary, main_df
        
    except Exception as e:
        logger.error(f"Error extracting Excel data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [], pd.DataFrame()


def prepare_data_for_analysis(summary: List[Dict]) -> str:
    """
    Prepare data in a format suitable for OpenAI analysis
    """
    analysis_data = {
        'summary_metrics': summary,
        'chart_data': summary,
        'data_points': len(summary) 
    }
    
    # Convert to readable text format
    text_data = "summary DATA:\n"
    for item in summary[:10]:  # Limit to first 10 items
        text_data += f"- {item}\n"
    
    return text_data


def create_insights_prompt(summary: List[Dict], report_title: str, brand_name: str, custom_prompt: str = None) -> str:
    """
    Create a comprehensive prompt for OpenAI insights generation
    """
    base_prompt = f"""
    Analyze the following media monitoring data and provide comprehensive insights:

    DATA:
    {summary}

    Please provide:
    1. Executive Summary (2-3 paragraphs)
    2. Key Performance Indicators Analysis
    3. Trend Analysis
    4. Brand Performance Insights
    5. Competitive Landscape Observations
    6. Recommendations for Future Actions
    7. Top 5 Key Messages
    8. Top 5 Suggested Communication Channels
    
    Don't:
    1. In the bullet points give it as list, don't mention number or alphabets or any type of symbol
    

    Brand Context: {brand_name}
    Report Focus: {report_title}

    Format your response as:
    EXECUTIVE SUMMARY: [summary]
    KPIs: [analysis]
    TRENDS: [trends]
    BRAND INSIGHTS: [insights]
    COMPETITIVE: [observations]
    RECOMMENDATIONS: [recommendations]
    KEY MESSAGES: [list of 5 key messages]
    SUGGESTED CHANNELS: [list of 5 channels]
    """
    
    if custom_prompt:
        base_prompt += f"\n\nAdditional Context: {custom_prompt}"
    
    return base_prompt


def parse_insights_response(insights_text: str) -> Dict[str, Any]:
    """
    Parse OpenAI response into structured format
    """
    try:
        # Extract different sections
        sections = {
            'insights': insights_text,
            'key_messages': [],
            'suggested_channels': []
        }
        
        # Extract key messages
        if 'KEY MESSAGES:' in insights_text:
            key_messages_section = insights_text.split('KEY MESSAGES:')[1].split('SUGGESTED CHANNELS:')[0]
            raw_lines = [ln.strip() for ln in key_messages_section.split('\n') if ln.strip()]
            key_messages = []
            for ln in raw_lines:
                # Accept bullet formats like '- ...', '* ...', '1. ...'
                cleaned = ln.lstrip('-* ').strip()
                if cleaned and not cleaned.upper().startswith('KEY MESSAGES'):
                    key_messages.append(cleaned)
            sections['key_messages'] = key_messages[:5]
        
        # Extract suggested channels
        if 'SUGGESTED CHANNELS:' in insights_text:
            channels_section = insights_text.split('SUGGESTED CHANNELS:')[1]
            raw_lines = [ln.strip() for ln in channels_section.split('\n') if ln.strip()]
            channels = []
            for ln in raw_lines:
                cleaned = ln.lstrip('-* ').strip()
                if cleaned and not cleaned.upper().startswith('SUGGESTED CHANNELS'):
                    channels.append(cleaned)
            sections['suggested_channels'] = channels[:5]
        
        return sections
        
    except Exception as e:
        logger.error(f"Error parsing insights: {e}")
        return {
            'insights': insights_text,
            'key_messages': [],
            'suggested_channels': []
        }


def get_chart_images_from_directory() -> List[Tuple[str, str]]:
    """
    Get chart images from assets directory using similar logic to get_brand_logo_path
    Returns list of (title, filepath) tuples
    """
    charts = []
    
    try:
        # Primary location: assets directory (same as logo)
        assets_dir = os.path.join(os.getcwd(), "assets")
        
        # Fallback location: uploads/image directory
        uploads_dir = os.path.join(os.getcwd(), "uploads", "image")
        
        # Try assets directory first
        directories_to_check = [assets_dir]
        if os.path.exists(uploads_dir):
            directories_to_check.append(uploads_dir)
        
        # Get all image files from directories
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']
        image_files = []
        
        for directory in directories_to_check:
            if os.path.exists(directory):
                for extension in image_extensions:
                    image_files.extend(glob.glob(os.path.join(directory, extension)))
                    image_files.extend(glob.glob(os.path.join(directory, extension.upper())))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        # Filter out logo files (we don't want to include them as charts)
        chart_files = [
            f for f in image_files 
            if not os.path.basename(f).lower().startswith('logo') 
            and not os.path.basename(f).lower().endswith('logo.png')
            and not os.path.basename(f).lower().endswith('logo.jpg')
        ]
        
        # Sort files for consistent ordering
        chart_files.sort()
        
        for image_path in chart_files:
            try:
                # Get a clean filename for the title
                filename = os.path.basename(image_path)
                # Remove extension and clean up the name
                chart_title = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
                
                # Store the file path (same approach as logo)
                charts.append((chart_title, image_path))
                logger.info(f"Successfully loaded chart: {chart_title} from {image_path}")
                
            except Exception as e:
                logger.warning(f"Could not load image {image_path}: {e}")
                continue
        
        if not charts:
            logger.info("No chart images found in assets or uploads/image directories")
        else:
            logger.info(f"Found {len(charts)} chart images")
        
    except Exception as e:
        logger.error(f"Error loading chart images from directory: {e}")
    
    return charts


def get_chart_description(chart_title: str) -> str:
    """
    Generate a description for each chart based on its title with graph type labeling
    """
    title_lower = chart_title.lower()
    
    # Enhanced descriptions with graph type identification
    if 'media distribution' in title_lower:
        return "📊 BAR CHART: This chart shows the distribution of media mentions over time, providing insights into media coverage patterns and trends."
    elif 'media type count' in title_lower:
        return "📈 COLUMN CHART: This visualization displays the count of different media types, helping identify which media channels are most active."
    elif 'platform value comparison' in title_lower:
        return "📊 COMPARISON CHART: This chart compares the value across different platforms, enabling analysis of platform performance and ROI."
    elif 'platform count over time' in title_lower:
        return "📈 LINE CHART: This graph tracks platform activity over time, showing trends in platform usage and engagement."
    elif 'platform value over time' in title_lower:
        return "📈 TREND CHART: This visualization shows how platform values change over time, revealing growth patterns and seasonal trends."
    elif 'sentiment count' in title_lower:
        return "🥧 PIE CHART: This chart analyzes sentiment distribution, providing insights into public opinion and brand perception."
    elif 'sentiment' in title_lower and 'analysis' in title_lower:
        return "📊 SENTIMENT ANALYSIS CHART: This visualization provides comprehensive sentiment analysis across different metrics and time periods."
    elif 'pr value progression' in title_lower:
        return "📈 PROGRESSION CHART: This graph tracks PR value progression, showing the cumulative impact of public relations efforts."
    elif 'pr value by period' in title_lower:
        return "📊 PERIOD ANALYSIS CHART: This chart breaks down PR value by time periods, enabling analysis of campaign effectiveness over different intervals."
    elif '3d' in title_lower or 'three' in title_lower:
        return "🎯 3D VISUALIZATION: This three-dimensional chart provides multi-dimensional analysis of complex data relationships."
    elif 'scatter' in title_lower:
        return "🔍 SCATTER PLOT: This chart shows correlations and relationships between different data points."
    elif 'heatmap' in title_lower:
        return "🔥 HEATMAP: This visualization shows data density and patterns through color intensity."
    else:
        return "📊 DATA VISUALIZATION: This chart provides valuable insights into the data analysis, supporting strategic decision-making and performance evaluation."


def detect_graph_type(chart_title: str) -> str:
    """
    Detect and return the type of graph based on chart title
    """
    title_lower = chart_title.lower()
    
    # Graph type detection logic
    if any(keyword in title_lower for keyword in ['bar', 'column', 'histogram']):
        return "Bar Chart"
    elif any(keyword in title_lower for keyword in ['line', 'trend', 'progression', 'over time']):
        return "Line Chart"
    elif any(keyword in title_lower for keyword in ['pie', 'donut', 'distribution']):
        return "Pie Chart"
    elif any(keyword in title_lower for keyword in ['scatter', 'correlation']):
        return "Scatter Plot"
    elif any(keyword in title_lower for keyword in ['heatmap', 'heat map', 'density']):
        return "Heatmap"
    elif any(keyword in title_lower for keyword in ['3d', 'three', 'dimensional']):
        return "3D Visualization"
    elif any(keyword in title_lower for keyword in ['area', 'stacked']):
        return "Area Chart"
    elif any(keyword in title_lower for keyword in ['bubble', 'size']):
        return "Bubble Chart"
    else:
        return "Data Visualization"


def organize_charts_by_category(charts: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Organize charts into categories based on their titles for better presentation
    Enhanced to work with both generated charts and existing chart images
    """
    categories = {
        "Trend Analysis & Time Series": [],
        "Platform & Channel Analysis": [],
        "Sentiment & Engagement Analysis": [],
        "Value & ROI Analysis": [],
        "Correlation & Advanced Analytics": [],
        "Other Analytics": [],
        "Data Distribution & Performance": [],
    }
    
    for chart_title, chart_path in charts:
        title_lower = chart_title.lower()
        graph_type = detect_graph_type(chart_title)
        
        # Enhanced categorization with better keywords and emojis
        if any(keyword in title_lower for keyword in ['media', 'distribution', 'count', 'type', 'pie', 'bar']):
            categories["Data Distribution & Performance"].append((chart_title, chart_path))
        elif any(keyword in title_lower for keyword in ['trend', 'time', 'over time', 'progression', 'line', 'series']):
            categories["Trend Analysis & Time Series"].append((chart_title, chart_path))
        elif any(keyword in title_lower for keyword in ['platform', 'channel', 'comparison', 'performance']):
            categories["Platform & Channel Analysis"].append((chart_title, chart_path))
        elif any(keyword in title_lower for keyword in ['sentiment', 'engagement', 'emotion', 'feeling']):
            categories["Sentiment & Engagement Analysis"].append((chart_title, chart_path))
        elif any(keyword in title_lower for keyword in ['value', 'roi', 'revenue', 'cost', 'pr value', 'monetary']):
            categories["Value & ROI Analysis"].append((chart_title, chart_path))
        elif any(keyword in title_lower for keyword in ['correlation', 'matrix', 'heatmap', 'scatter', 'relationship']):
            categories["Correlation & Advanced Analytics"].append((chart_title, chart_path))
        else:
            categories["Other Analytics"].append((chart_title, chart_path))
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def extract_insight_sections(insights_text: str) -> Dict[str, str]:
    """Extract different sections from the insights text"""
    sections = {}
    
    # Define section patterns
    section_patterns = {
        'KPIs Analysis': ['KPIs:', 'KPI:', 'Key Performance Indicators:'],
        'Trend Analysis': ['TRENDS:', 'Trends:', 'TREND ANALYSIS:'],
        'Brand Insights': ['BRAND INSIGHTS:', 'Brand Insights:', 'BRAND PERFORMANCE:'],
        'Competitive Analysis': ['COMPETITIVE:', 'Competitive:', 'COMPETITIVE LANDSCAPE:'],
        'Recommendations': ['RECOMMENDATIONS:', 'Recommendations:', 'RECOMMENDED ACTIONS:']
    }
    
    for section_name, patterns in section_patterns.items():
        for pattern in patterns:
            if pattern in insights_text:
                # Extract content after the pattern
                start_idx = insights_text.find(pattern) + len(pattern)
                # Find the next section or end of text
                next_section_idx = len(insights_text)
                for other_patterns in section_patterns.values():
                    for other_pattern in other_patterns:
                        if other_pattern != pattern and other_pattern in insights_text[start_idx:]:
                            idx = insights_text.find(other_pattern, start_idx)
                            if idx < next_section_idx:
                                next_section_idx = idx
                
                content = insights_text[start_idx:next_section_idx].strip()
                if content:
                    sections[section_name] = content
                break
    
    return sections


def create_decorative_line():
    """Create a decorative line element"""
    drawing = Drawing(400, 2)
    line = Line(0, 1, 400, 1)
    line.strokeColor = colors.HexColor('#1F4E78')
    line.strokeWidth = 2
    drawing.add(line)
    return drawing


def create_footer():
    """Create a footer with branding"""
    from reportlab.lib.styles import getSampleStyleSheet
    styles = getSampleStyleSheet()
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#999999'),
        fontName='Helvetica'
    )
    
    footer_text = f"Generated by Rantau Media Analytics | {datetime.datetime.now().strftime('%Y')}"
    return Paragraph(footer_text, footer_style)


def download_image_from_url(url: str) -> io.BytesIO:
    """
    Download image from URL (Cloudinary or any HTTP URL) and return as BytesIO
    """
    try:
        import requests
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return io.BytesIO(response.content)
    except Exception as e:
        logger.error(f"Error downloading image from URL {url}: {e}")
        raise


def extract_chart_urls(charts: List[Tuple[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Extract and filter chart URLs for pie charts, bar charts, and scatter plots only.
    Returns a dictionary with chart type as key and URL as value.
    
    Args:
        charts: List of (title, url_or_path) tuples from chart generation
    
    Returns:
        Dictionary with keys: 'pie_charts', 'bar_charts', 'scatter_plots'
        Each key contains a list of dicts with 'title' and 'url'
    """
    chart_urls = {
        'pie_charts': [],
        'bar_charts': [],
        'scatter_plots': []
    }
    
    for chart_title, chart_path in charts:
        # Only process Cloudinary URLs (not local paths)
        if not chart_path or not (chart_path.startswith('http://') or chart_path.startswith('https://')):
            continue
        
        title_lower = chart_title.lower()
        
        # Detect chart type and categorize
        if 'pie' in title_lower or 'share' in title_lower:
            chart_urls['pie_charts'].append({
                'title': chart_title,
                'url': chart_path
            })
        elif 'bar' in title_lower or 'distribution' in title_lower or 'count' in title_lower:
            chart_urls['bar_charts'].append({
                'title': chart_title,
                'url': chart_path
            })
        elif 'scatter' in title_lower or 'correlation' in title_lower or 'vs' in title_lower:
            chart_urls['scatter_plots'].append({
                'title': chart_title,
                'url': chart_path
            })
    
    return chart_urls


def get_brand_logo_path(brand_name: str = None) -> str:
    """
    Get the brand logo path, with fallback to default logo
    """
    # Try brand-specific logo first (Gentari)
    if brand_name and 'gentari' in brand_name.lower():
        gentari_logo_path = os.path.join(os.getcwd(), "assets", "gentari-logo.png")
        if os.path.exists(gentari_logo_path):
            return gentari_logo_path
    
    # Try brand-specific logo (generic)
    if brand_name:
        brand_logo_path = os.path.join(os.getcwd(), "assets", f"{brand_name.lower()}-logo.png")
        if os.path.exists(brand_logo_path):
            return brand_logo_path
    
    # Fallback to default logo
    default_logo_path = os.path.join(os.getcwd(), "assets", "logo-dark.png")
    if os.path.exists(default_logo_path):
        return default_logo_path
    
    # Another fallback
    uploads_logo_path = os.path.join(os.getcwd(), "uploads", "image", "logo-dark.png")
    if os.path.exists(uploads_logo_path):
        return uploads_logo_path
    
    # Final fallback
    return os.path.join(os.getcwd(), "uploads", "image", "logo.png")


def create_pdf_report(summary: List[Dict], insights: Dict[str, Any], 
                     report_title: str, brand_name: str, charts: List[Tuple[str, str]] = None) -> io.BytesIO:
    """
    Create a beautiful PDF report using ReportLab with brand logo and improved formatting
    """
    if not REPORTLAB_AVAILABLE:
        raise Exception("ReportLab library not available. Please install reportlab for PDF generation.")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=100, bottomMargin=50)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles with better formatting
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        spaceAfter=40,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1F4E78'),
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=15,
        spaceBefore=20,
        textColor=colors.HexColor('#1F4E78'),
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.HexColor('#1F4E78'),
        borderPadding=8,
        backColor=colors.HexColor('#F0F4F8')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=16,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.HexColor('#2E5B8A'),
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        leading=14
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        leftIndent=20,
        fontName='Helvetica',
        leading=14
    )
    
    # Build content
    story = []
    
    # Add brand logo with smart logo detection
    logo_path = get_brand_logo_path(brand_name)
    if os.path.exists(logo_path):
        try:
            logo = Image(logo_path, width=120, height=60)
            logo.hAlign = 'CENTER'
            story.append(logo)
            story.append(Spacer(1, 20))
            logger.info(f"Successfully loaded brand logo: {logo_path}")
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")
    else:
        logger.warning(f"Logo not found at: {logo_path}")
    
    # Title page with better formatting
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 30))
    
    # Brand and date info with better styling
    brand_info_style = ParagraphStyle(
        'BrandInfo',
        parent=styles['Normal'],
        fontSize=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#666666'),
        fontName='Helvetica'
    )
    
    story.append(Paragraph(f"<b>Brand:</b> {brand_name}", brand_info_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", brand_info_style))
    story.append(Spacer(1, 30))
    
    # Add a decorative line
    story.append(create_decorative_line())
    story.append(PageBreak())
    
    # Executive Summary with improved formatting
    story.append(Paragraph("Executive Summary", heading_style))
    if insights.get('insights'):
        # Split insights into paragraphs for better readability
        insight_paragraphs = insights['insights'].split('\n\n')
        for paragraph in insight_paragraphs:
            if paragraph.strip():
                story.append(Paragraph(paragraph.strip(), body_style))
                story.append(Spacer(1, 8))
    story.append(Spacer(1, 25))
    
    # Key Messages with better formatting
    if insights.get('key_messages'):
        story.append(Paragraph("Key Messages", heading_style))
        story.append(Spacer(1, 10))
        for i, message in enumerate(insights['key_messages'], 1):
            story.append(Paragraph(f"<b>{i}.</b> {message}", bullet_style))
        story.append(Spacer(1, 25))
    
    # Suggested Channels with better formatting
    if insights.get('suggested_channels'):
        story.append(Paragraph("Recommended Communication Channels", heading_style))
        story.append(Spacer(1, 10))
        for i, channel in enumerate(insights['suggested_channels'], 1):
            story.append(Paragraph(f"<b>{i}.</b> {channel}", bullet_style))
        story.append(Spacer(1, 25))
    
    # Add insights sections if available
    if insights.get('insights'):
        insight_sections = extract_insight_sections(insights['insights'])
        
        for section_title, section_content in insight_sections.items():
            if section_content.strip():
                story.append(Paragraph(section_title, heading_style))
                story.append(Spacer(1, 10))
                story.append(Paragraph(section_content.strip(), body_style))
                story.append(Spacer(1, 20))
    
    # Add charts section (from uploads/image directory)
    if charts is not None and len(charts) > 0:
        story.append(Paragraph("Data Visualization - Analytics Charts", heading_style))
        story.append(Spacer(1, 15))
        
        # Group charts by category for better organization
        chart_categories = organize_charts_by_category(charts)
        
        for category, category_charts in chart_categories.items():
            if category_charts:
                # Add category header
                story.append(Paragraph(category, subheading_style))
                story.append(Spacer(1, 10))
                
                # Add charts in this category
                for chart_title, chart_path in category_charts:
                    try:
                        logger.info(f"Processing chart: {chart_title}")
                        
                        # Detect graph type and add it to the title
                        graph_type = detect_graph_type(chart_title)
                        
                        # Add chart title with graph type label
                        story.append(Paragraph(f"• {chart_title} ({graph_type})", subheading_style))
                        story.append(Spacer(1, 8))
                        
                        # Create and add chart image - support both Cloudinary URLs and local paths
                        try:
                            # Check if chart_path is a URL (Cloudinary) or local file path
                            if chart_path and (chart_path.startswith('http://') or chart_path.startswith('https://')):
                                # Download from Cloudinary URL
                                logger.info(f"Downloading chart from Cloudinary URL: {chart_path}")
                                img_buffer = download_image_from_url(chart_path)
                                chart_img = Image(img_buffer, width=6*inch, height=4*inch)
                            elif chart_path and os.path.exists(chart_path):
                                # Use local file path (backward compatibility)
                                logger.info(f"Using local chart file: {chart_path}")
                                chart_img = Image(chart_path, width=6*inch, height=4*inch)
                            else:
                                raise FileNotFoundError(f"Chart not found: {chart_path}")
                            
                            chart_img.hAlign = 'CENTER'
                            logger.info(f"Successfully created image object for {chart_title}")
                            story.append(chart_img)
                            story.append(Spacer(1, 8))
                            
                            # Add description below the chart
                            chart_description = get_chart_description(chart_title)
                            description_style = ParagraphStyle(
                                'ChartDescription',
                                parent=styles['Normal'],
                                fontSize=10,
                                spaceAfter=12,
                                alignment=TA_CENTER,
                                textColor=colors.HexColor('#666666'),
                                fontName='Helvetica-Oblique',
                                leftIndent=20,
                                rightIndent=20
                            )
                            story.append(Paragraph(chart_description, description_style))
                            story.append(Spacer(1, 15))
                            
                        except Exception as img_error:
                            logger.warning(f"Failed to create image object for {chart_title}: {img_error}")
                            # Fallback: Add a placeholder text for the chart
                            story.append(Paragraph(f"📊 {chart_title}", body_style))
                            story.append(Paragraph("Chart image available but could not be embedded in PDF", body_style))
                            story.append(Spacer(1, 10))
                            
                    except Exception as e:
                        logger.error(f"Error adding chart {chart_title}: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        story.append(Paragraph(f"Error loading {chart_title}", body_style))
                        story.append(Spacer(1, 10))
                
                # Add spacing between categories
                story.append(Spacer(1, 20))
    
    # Add footer with page numbers and branding
    story.append(create_footer())
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


async def generate_qualitative_insights(summary_data: List[Dict], charts_data: List[Dict] = None, 
                                      report_title: str = None, brand_name: str = None, custom_prompt: str = None) -> Dict[str, Any]:
    """
    Generate qualitative insights using OpenAI API
    """
    try:
        # Prepare data for OpenAI analysis
        prepared_data = prepare_data_for_analysis(summary_data)
        
        # Create prompt for OpenAI (pass the prepared string data)
        prompt = create_insights_prompt(summary_data, report_title or "Media Report", brand_name or "Brand", custom_prompt)
        
        # Enhance prompt with prepared data
        if prepared_data:
            prompt = f"{prompt}\n\nPREPARED DATA:\n{prepared_data}"
        
        # Call OpenAI API (config.OPENAI_API_KEY is loaded from .env file)
        logger.info(f"Calling OpenAI API with model: gpt-4o, prompt length: {len(prompt)}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert media analyst and business strategist. Provide detailed, actionable insights based on the data provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        insights_text = response.choices[0].message.content
        logger.info(f"Successfully received insights from OpenAI, length: {len(insights_text)}")
        
        # Parse insights into structured format
        parsed_insights = parse_insights_response(insights_text)
        logger.info(f"Successfully parsed insights: {len(parsed_insights.get('key_messages', []))} key messages, {len(parsed_insights.get('suggested_channels', []))} channels")
        
        return parsed_insights
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error generating insights: {e}")
        logger.error(f"Full traceback: {error_details}")
        
        # Return more informative error message
        error_message = f"Unable to generate insights at this time. Error: {str(e)}"
        if "API key" in str(e) or "authentication" in str(e).lower():
            error_message = "OpenAI API authentication failed. Please check your API key configuration."
        elif "rate limit" in str(e).lower():
            error_message = "OpenAI API rate limit exceeded. Please try again later."
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            error_message = "Network error connecting to OpenAI API. Please check your internet connection."
        
        return {
            'insights': error_message,
            'key_messages': [],
            'suggested_channels': []
        }


async def generate_pdf_report_from_excel(excel_bytes: bytes, report_title: str, brand_name: str, 
                                       include_charts: bool = True, include_insights: bool = True, 
                                       custom_prompt: str = None) -> Tuple[io.BytesIO, Dict[str, Any], Dict[str, List[Dict[str, str]]]]:
    """
    Generate a beautiful PDF report from Excel data with OpenAI insights and generated charts
    Returns: (pdf_buffer, insights, chart_urls)
    """
    if not REPORTLAB_AVAILABLE:
        raise Exception("ReportLab library not available. Please install reportlab for PDF generation.")
    
    try:
        # Extract data from Excel file (returns summary list and main DataFrame)
        summary, main_df = extract_excel_data(excel_bytes)
        
        # Generate OpenAI insights
        insights = {}
        if include_insights:
            # Extract charts_data if available (can be empty list if no charts)
            charts_data = []
            try:
                # Try to extract chart data from summary if available
                charts_data = summary if isinstance(summary, list) else []
            except Exception:
                charts_data = []
            
            insights = await generate_qualitative_insights(summary, charts_data, report_title, brand_name, custom_prompt)
        
        # Generate charts using seaborn/matplotlib (saved to assets directory)
        charts = []
        if include_charts and MATPLOTLIB_AVAILABLE:
            charts = generate_charts_from_data(summary, brand_name, main_df)
            if charts:
                logger.info(f"Generated {len(charts)} charts using seaborn/matplotlib in assets directory")
            else:
                # Fallback to existing chart images from assets directory
                charts = get_chart_images_from_directory()
                logger.info(f"Using {len(charts)} existing chart images from assets directory")
        elif include_charts:
            # Fallback to existing chart images if matplotlib not available
            charts = get_chart_images_from_directory()
            logger.info(f"Matplotlib not available, using {len(charts)} existing chart images from assets directory")
        
        # Extract chart URLs for pie, bar, and scatter plots only
        chart_urls = extract_chart_urls(charts)
        
        # Create PDF report
        pdf_buffer = create_pdf_report(summary, insights, report_title, brand_name, charts)
        
        return pdf_buffer, insights, chart_urls
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise Exception(f"Error generating PDF report: {str(e)}")

def extract_drawing_as_image(drawing, sheet):
    """
    Extract a drawing object from Excel sheet as an image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a blank image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to get a default font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw drawing title
        draw.text((50, 50), f"Drawing from {sheet.title}", fill='black', font=font)
        
        # Draw some basic representation
        draw.rectangle([50, 100, 750, 550], outline='black', width=2)
        draw.text((100, 120), "Drawing/Chart extracted from Excel", fill='green', font=font)
        
        return img
        
    except Exception as e:
        logger.warning(f"Could not create drawing image: {e}")
    
    return None


def extract_chart_as_image(chart, sheet):
    """
    Extract a chart from Excel sheet as an image
    """
    try:
        # This is a simplified approach - in practice, you might need more sophisticated chart extraction
        # For now, we'll try to create a representation of the chart data
        
        # Get chart data if available
        if hasattr(chart, 'data') and chart.data:
            # Create a simple representation
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            
            # Create a blank image
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to get a default font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw chart title
            draw.text((50, 50), f"Chart from {sheet.title}", fill='black', font=font)
            
            # Draw some basic chart representation
            draw.rectangle([50, 100, 750, 550], outline='black', width=2)
            draw.text((100, 120), "Chart data extracted from Excel", fill='blue', font=font)
            
            return img
            
    except Exception as e:
        logger.warning(f"Could not create chart image: {e}")
    
    return None

def fig_to_base64(fig):
    """Convert Plotly figure to base64 encoded image"""
    try:
        # Convert to static image
        img_bytes = pio.to_image(fig, format="png", width=800, height=600, scale=2)
        img_base64 = base64.b64encode(img_bytes).decode()
        return img_base64
    except Exception as e:
        logger.error(f"Error converting figure to base64: {e}")
        return None


def create_chart_image_from_base64(base64_string, title):
    """
    Create a ReportLab Image object from base64 string with optimized sizing for charts
    """
    try:
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available, cannot create chart image")
            return None
            
        # Decode base64 to bytes
        image_data = base64.b64decode(base64_string)
        
        # Create ImageReader from bytes
        img_reader = ImageReader(io.BytesIO(image_data))
        
        # Get original dimensions
        img_width, img_height = img_reader.getSize()
        
        logger.info(f"Original image dimensions for {title}: {img_width} x {img_height}")
        
        # Calculate optimal size for PDF (max width 6 inches, maintain aspect ratio)
        max_width = 6 * inch
        max_height = 4 * inch
        
        # Calculate scale factor to fit within max dimensions
        width_scale = max_width / img_width if img_width > max_width else 1.0
        height_scale = max_height / img_height if img_height > max_height else 1.0
        
        # Use the smaller scale factor to ensure image fits within bounds
        scale_factor = min(width_scale, height_scale)
        
        new_width = img_width * scale_factor
        new_height = img_height * scale_factor
        
        logger.info(f"Scaled image dimensions for {title}: {new_width} x {new_height}")
        
        # Create ReportLab Image with explicit sizing
        img = Image(img_reader, width=new_width, height=new_height)
        
        # Center the image
        img.hAlign = 'CENTER'
        
        logger.info(f"Successfully created ReportLab image for {title}")
        return img
        
    except Exception as e:
        logger.error(f"Error creating chart image from base64 for {title}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def create_graph_image_from_base64(base64_string, title):
    """Create a ReportLab Image from base64 string"""
    try:
        # Decode base64 to bytes
        img_data = base64.b64decode(base64_string)
        
        # Create temporary file
        temp_file = io.BytesIO(img_data)
        
        # Create ReportLab Image
        img = Image(temp_file, width=400, height=300)
        img.hAlign = 'CENTER'
        
        return img
    except Exception as e:
        logger.error(f"Error creating image from base64: {e}")
        return None
