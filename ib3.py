import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# --- Custom Color Palette ---
CUSTOM_COLORS = {
    'Biscay': '#183F5E',    # Darkest Blue (Negative/Strong emphasis)
    'Edgewater': '#BADED7', # Lightest Blue-Green (Positive)
    'Neptune': '#73B4B6',   # Mid Blue-Green (Neutral/Service-related)
    'Mariner': '#267EBB'    # Medium Blue (Sales-related/General)
}

# Mapping for Sentiment labels
SENTIMENT_COLOR_MAP = {
    'Positive': CUSTOM_COLORS['Edgewater'], # Lightest for positive
    'Neutral': CUSTOM_COLORS['Neptune'],    # Mid-tone for neutral
    'Negative': CUSTOM_COLORS['Biscay'],    # Darkest for negative
    'Unknown': '#CCCCCC' # A neutral grey for unknown/unspecified (though 'Unknown' will be filtered for plots)
}

# Mapping for Agent Rating
AGENT_RATING_COLOR_MAP = {
    'High': CUSTOM_COLORS['Edgewater'],  # Positive
    'Medium': CUSTOM_COLORS['Neptune'],  # Neutral
    'Low': CUSTOM_COLORS['Biscay'],     # Negative
    'Unknown': '#CCCCCC' # (will be filtered for plots)
}

# --- Page Configuration ---
st.set_page_config(
    page_title="Inbound Call Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
st.markdown(f"""
<style>
/* Main Block Container */
.block-container {{
    padding-top: 1rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem;
}}

/* Metric Card Styling */
div[data-testid="metric-container"] {{
    background-color: #f0f2f6;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
}}

div[data-testid="stMetricLabel"] p {{
    font-size: 16px; /* Label font size, as requested */
    font-weight: bold;
    color: #555555; /* Darker label color */
}}

div[data-testid="stMetricValue"] {{
    font-size: 36px; /* Value font size */
    font-weight: bold;
    color: {CUSTOM_COLORS['Mariner']}; /* Use a color from the palette */
}}

/* Header Styling */
h1 {{
    color: {CUSTOM_COLORS['Biscay']}; text-align: center; font-size: 36px;
}}
h2 {{
    color: {CUSTOM_COLORS['Mariner']}; border-bottom: 2px solid {CUSTOM_COLORS['Neptune']};
    padding-bottom: 5px; margin-top: 40px; font-size: 28px;
}}
h5 {{
    color: {CUSTOM_COLORS['Biscay']}; margin-top: 15px; margin-bottom: 5px; font-size: 20px;
}}

/* General paragraph text size */
p, ul, ol, li, div.stMarkdown, div.stInfo, div.stWarning {{
    font-size: 16px !important;
}}

/* Business Question Strip Styling */
.question-strip {{
    background-color: #E6EEF6; padding: 15px 20px; margin-top: 25px; margin-bottom: 15px;
    border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.question-strip h3 {{
    color: #333333; margin: 0; font-size: 20px; font-weight: bold; text-align: center; /* Centered */
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data
def load_data(file):
    """Loads and preprocesses data from an uploaded CSV file."""
    try:
        string_data = io.StringIO(file.getvalue().decode('utf-8'))
        df = pd.read_csv(string_data)
        df.columns = df.columns.str.strip()

        if 'CallID' in df.columns:
            df = df[df['CallID'] != 'PARSE_FAILED']

        nan_fill_defaults = {
            'UserIntentToBuy': 'Not Specified', 'CallSentiment': 'Unknown',
            'Major Purchase Barrier Theme': 'Not Specified', 'Top3Themes': 'Not Specified',
            'AgentNextAction': 'Unknown', 'AgentRating': 'Unknown', 'Call Type': 'Unknown'
        }
        for col, default_val in nan_fill_defaults.items():
            if col in df.columns:
                # Replace 'nan', 'N/A', and empty strings with default_val
                df[col] = df[col].astype(str).replace(['nan', 'N/A', ''], default_val, regex=False)
                df[col] = df[col].fillna(default_val).str.strip()
            else:
                df[col] = default_val
        return df
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}. Please ensure it's a valid CSV.")
        return pd.DataFrame()

def find_col(df_columns, possible_names):
    """Finds a column name, case-insensitively."""
    df_columns_lower = {c.lower(): c for c in df_columns}
    for name in possible_names:
        if name.lower() in df_columns_lower:
            return df_columns_lower[name.lower()]
    return None

def ordered_intent_counts(df_filtered, col):
    """Orders user intent counts by a predefined order and includes percentages."""
    order = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Not Specified']
    if df_filtered.empty or col not in df_filtered.columns:
        return pd.DataFrame()

    counts = df_filtered[col].value_counts().reindex(order, fill_value=0)
    out_df = counts.reset_index()
    out_df.columns = ['intent', 'count']
    out_df = out_df[out_df['count'] > 0] # Filter out intents with zero count
    total = out_df['count'].sum()
    if total > 0:
        out_df['pct'] = (out_df['count'] / total * 100).round(1)
        out_df['text_label'] = out_df.apply(lambda row: f"{row['count']} ({row['pct']:.1f}%)", axis=1)
    else:
        out_df['pct'], out_df['text_label'] = 0, "0 (0.0%)"
    
    out_df['intent'] = pd.Categorical(out_df['intent'], categories=order, ordered=True)
    out_df = out_df.sort_values('intent')
    return out_df

def display_placeholder(message, height_px=250):
    """Displays an info message with a specific height."""
    st.info(message)
    st.markdown(f"<div style='height: {height_px-80}px;'></div>", unsafe_allow_html=True) # Adjust 80px for st.info height

def display_business_question(question_text):
    """Displays a business question in a styled strip."""
    st.markdown(f'<div class="question-strip"><h3>{question_text}</h3></div>', unsafe_allow_html=True)


# ---------------------------
# UI - Sidebar
# ---------------------------
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Inbound Call Analysis CSV", type=["csv"])

st.sidebar.markdown("""
**Expected Columns:**
- `Call Type` (Sale / Service)
- `UserIntentToBuy` (Very Low, Low, Medium, High, Very High)
- `CallSentiment` (Positive, Neutral, Negative)
- `Major Purchase Barrier Theme`
- `Top3Themes` (comma-separated, e.g., "Price, Discount, Delivery")
- `AgentNextAction` (Yes / No)
- `AgentRating` (High, Medium, Low)
- `City` (for geographical filtering)
""")

if not uploaded_file:
    st.title("Inbound Call Analytics Dashboard")
    st.info("Upload your CSV file from the sidebar to begin analysis.")
    st.stop()

# ---------------------------
# Load & Prepare Data
# ---------------------------
df = load_data(uploaded_file)
if df.empty: st.stop()

# Find columns
col_calltype = find_col(df.columns, ['Call Type'])
col_intent = find_col(df.columns, ['UserIntentToBuy'])
col_sentiment = find_col(df.columns, ['CallSentiment'])
col_barrier = find_col(df.columns, ['Major Purchase Barrier Theme'])
col_top3themes = find_col(df.columns, ['Top3Themes'])
col_nextaction = find_col(df.columns, ['AgentNextAction'])
col_agent_rating = find_col(df.columns, ['AgentRating'])
col_city = find_col(df.columns, ['City'])

if not col_calltype:
    st.error("Mandatory column 'Call Type' not found. Please check your CSV."); st.stop()

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("Filter Data")
filtered_df = df.copy()
if col_city:
    all_cities = ['All'] + sorted(df[col_city].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox('Select City', options=all_cities)
    if selected_city != 'All':
        filtered_df = df[df[col_city] == selected_city]

if filtered_df.empty: st.warning("No data available for the selected filters."); st.stop()

# Create dedicated dataframes for Sales and Service
sales_df = filtered_df[filtered_df[col_calltype].str.lower() == 'sale']
service_df = filtered_df[filtered_df[col_calltype].str.lower() == 'service']

# ---------------------------
# Main Dashboard Layout
# ---------------------------
st.title("Inbound Call Analytics Dashboard")
st.markdown("Insights into customer interactions from inbound calls – sales and service performance at a glance.")

# --- Overall KPIs (Top Metric) ---
total_calls = len(filtered_df)
num_sales = len(sales_df)
num_service = len(service_df)

col_metric, col_spacer = st.columns([1, 3]) # Use a spacer for layout

with col_metric:
    st.metric(label="Total Calls Analysed", value=f"{total_calls:,}", delta=f"Sales {num_sales:,} • Service {num_service:,}")

# Add a separator
st.markdown("---")

# --- Overall Insights ---
st.header("Overall Insights")
display_business_question("Q1. What type of Calls do we get?")
call_type_counts = filtered_df[col_calltype].value_counts().reset_index()
call_type_counts.columns = ['Call Type', 'Count']
total_calls_q1 = call_type_counts['Count'].sum()
if total_calls_q1 > 0:
    call_type_counts['Percentage'] = (call_type_counts['Count'] / total_calls_q1 * 100).round(1)
    call_type_counts['text_label'] = call_type_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

    fig = px.bar(call_type_counts, x='Call Type', y='Count', text='text_label',
                 color='Call Type', color_discrete_map={'Sale': CUSTOM_COLORS['Mariner'], 'Service': CUSTOM_COLORS['Neptune']})
    fig.update_traces(textposition='outside', cliponaxis=False, width=0.5, textfont_size=16)
    fig.update_layout(height=350, showlegend=False, yaxis_title='Number of Calls',
                      xaxis_title="", # Removed "Call Type" as it's clear from categories
                      yaxis_range=[0, call_type_counts['Count'].max() * 1.2],
                      xaxis_title_font_size=16, yaxis_title_font_size=16,
                      xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                      bargap=1.0) # Apply bargap here
    st.plotly_chart(fig, use_container_width=True)
else:
    display_placeholder("No 'Call Type' data to display.")

# Add a separator
st.markdown("---")

# --- Sales Insights ---
st.header("Sales Insights")

display_business_question("Q2. What is the User Intent to Buy in Sales calls?")
if col_intent and not sales_df.empty:
    intent_df = ordered_intent_counts(sales_df, col_intent) # This already generates text_label
    if not intent_df.empty:
        fig = px.bar(intent_df, x='intent', y='count', text='text_label',
                     labels={'intent': 'Intent Level', 'count': 'Number of Calls'},
                     color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
        fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
        fig.update_layout(height=400, yaxis_title='Number of Calls', xaxis_title='Intent Level',
                          yaxis_range=[0, intent_df['count'].max() * 1.25],
                          xaxis_title_font_size=16, yaxis_title_font_size=16,
                          xaxis_tickfont_size=14, yaxis_tickfont_size=14
                    ) 
        st.plotly_chart(fig, use_container_width=True)
    else:
        display_placeholder("No 'UserIntentToBuy' data available for Sales calls.")
else:
    display_placeholder("Column 'UserIntentToBuy' not found or no Sales calls in data.")


display_business_question("Q3. What is the sentiment of Sales Calls, broken down by Intent?")
if col_intent and col_sentiment and not sales_df.empty:
    # Get raw counts first
    raw_counts_cross_tab = pd.crosstab(index=sales_df[col_intent], columns=sales_df[col_sentiment])
    
    # Get percentages (normalize by index means row-wise percentage)
    pct_cross_tab = pd.crosstab(index=sales_df[col_intent], columns=sales_df[col_sentiment], normalize='index').mul(100).round(1)
    
    sentiment_display_order = ['Positive', 'Neutral', 'Negative']
    intent_order_for_plot = ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
    
    # Reindex both tables to ensure consistent columns and row order
    raw_counts_cross_tab = raw_counts_cross_tab.reindex(index=[i for i in intent_order_for_plot if i in raw_counts_cross_tab.index], 
                                                        columns=sentiment_display_order, fill_value=0)
    pct_cross_tab = pct_cross_tab.reindex(index=[i for i in intent_order_for_plot if i in pct_cross_tab.index],
                                          columns=sentiment_display_order, fill_value=0)

    # Melt both into long format
    plot_df_counts = raw_counts_cross_tab.reset_index().melt(id_vars=col_intent, var_name='Sentiment', value_name='Count')
    plot_df_pct = pct_cross_tab.reset_index().melt(id_vars=col_intent, var_name='Sentiment', value_name='Percentage')

    # Merge to combine counts and percentages
    plot_df = pd.merge(plot_df_counts, plot_df_pct, on=[col_intent, 'Sentiment'])
    
    # Create the combined text label
    plot_df['text_label'] = plot_df.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)" if row['Count'] > 0 else "", axis=1)

    # Filter out rows where Percentage is 0 if text is empty (to avoid clutter if there's no data for that segment)
    plot_df = plot_df[plot_df['Count'] > 0] # Only plot segments that actually have calls

    fig = px.bar(plot_df, x=col_intent, y='Percentage', color='Sentiment', barmode='group',
                 text='text_label', # Use the new combined label
                 color_discrete_map=SENTIMENT_COLOR_MAP, 
                 category_orders={col_intent: intent_order_for_plot, "Sentiment": sentiment_display_order},
                 labels={'Percentage': 'Percentage of Calls', col_intent: 'User Intent to Buy', 'Count': 'Number of Calls'})
    fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=14) 
    fig.update_layout(height=500, yaxis_title='Sentiment Distribution (%)', legend_title_text='Sentiment', yaxis_range=[0, 115],
                      xaxis_title_font_size=16, yaxis_title_font_size=16,
                      xaxis_tickfont_size=14, yaxis_tickfont_size=14
                      ) 
    st.plotly_chart(fig, use_container_width=True)
else:
    display_placeholder("Cannot generate Intent vs. Sentiment chart. Check for 'UserIntentToBuy', 'CallSentiment', and Sales calls.")


display_business_question("Q4. What are the reasons why Sales Customers are not Converting?")
if col_barrier and not sales_df.empty:
    barrier_counts = sales_df[col_barrier].value_counts().reset_index()
    barrier_counts.columns = ['Barrier', 'Count']
    # Filter out non-descriptive barriers including 'Not Specified' and 'No Barrier / Info Unavailable'
    barrier_counts = barrier_counts[~barrier_counts['Barrier'].isin(['Not Specified', 'No Barrier / Info Unavailable'])]
    
    total_barriers_q4 = barrier_counts['Count'].sum()
    if total_barriers_q4 > 0:
        barrier_counts['Percentage'] = (barrier_counts['Count'] / total_barriers_q4 * 100).round(1)
        barrier_counts['text_label'] = barrier_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

        if not barrier_counts.empty:
            fig = px.bar(barrier_counts.head(10).sort_values('Count', ascending=True),
                         x='Count', y='Barrier', orientation='h', text='text_label', title="Top Purchase Barriers in Sales Calls",
                         color_discrete_sequence=[CUSTOM_COLORS['Biscay']])
            fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
            fig.update_layout(height=400, xaxis_title='Number of Mentions', yaxis_title='', xaxis_range=[0, barrier_counts['Count'].max() * 1.25],
                              xaxis_title_font_size=16, yaxis_title_font_size=16,
                              xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                              yaxis_automargin=True,
                              ) 
            st.plotly_chart(fig, use_container_width=True)
        else:
            display_placeholder("No significant purchase barrier data found for Sales calls.")
    else:
        display_placeholder("No significant purchase barrier data found for Sales calls.")
else:
    display_placeholder("Column 'Major Purchase Barrier Theme' not found or no Sales calls in data.")

display_business_question("Q5. What are the Talking Points in a Sales Call?")
if col_top3themes and not sales_df.empty:
    themes = sales_df[col_top3themes].str.split(',').explode().str.strip()
    # Filter out non-descriptive themes
    themes = themes.dropna()[~themes.isin(['Not Specified', 'No Barrier / Info Unavailable', 'Other'])]
    
    total_themes_q5 = themes.shape[0] # Total mentions
    if total_themes_q5 > 0:
        theme_counts = themes.value_counts().reset_index()
        theme_counts.columns = ['Theme', 'Count']
        theme_counts['Percentage'] = (theme_counts['Count'] / total_themes_q5 * 100).round(1)
        theme_counts['text_label'] = theme_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

        if not theme_counts.empty:
            fig = px.bar(theme_counts.head(10).sort_values('Count', ascending=True),
                         x='Count', y='Theme', orientation='h', text='text_label', title="Top 10 Talking Points in Sales Calls",
                         color_discrete_sequence=[CUSTOM_COLORS['Mariner']])
            fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
            fig.update_layout(height=450, xaxis_title='Number of Mentions', yaxis_title='', xaxis_range=[0, theme_counts['Count'].max() * 1.25],
                              xaxis_title_font_size=16, yaxis_title_font_size=16,
                              xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                              yaxis_automargin=True
                              )
            st.plotly_chart(fig, use_container_width=True)
        else:
            display_placeholder("No talking points data available for Sales calls.")
    else:
        display_placeholder("No talking points data available for Sales calls.")
else:
    display_placeholder("Column 'Top3Themes' not found or no Sales calls in data.")

# Sales - Q6 and Q7 in columns
col1, col2 = st.columns(2)
#with col1:
display_business_question("Q6. How many Sales Calls Require a Follow-up?")
if col_nextaction and not sales_df.empty:
        action_counts = sales_df[col_nextaction].value_counts()
        # Filter out 'Unknown'
        action_counts = action_counts[action_counts.index != 'Unknown']
        
        total_actions_q6 = action_counts.sum()
        if total_actions_q6 > 0:
            action_counts = action_counts.reset_index()
            action_counts.columns = ['Follow-up', 'Count']
            action_counts['Percentage'] = (action_counts['Count'] / total_actions_q6 * 100).round(1)
            action_counts['text_label'] = action_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

            if not action_counts.empty:
                fig = px.bar(action_counts, x='Follow-up', y='Count', text='text_label', color='Follow-up',
                             color_discrete_map={'Yes': CUSTOM_COLORS['Mariner'], 'No': CUSTOM_COLORS['Biscay']}) # No 'Unknown' in map
                fig.update_traces(textposition='outside', cliponaxis=False, width=0.4, textfont_size=16)
                fig.update_layout(height=400, title="Agent Follow-up for Sales", showlegend=False, yaxis_range=[0, action_counts['Count'].max() * 1.25],
                                  xaxis_title_font_size=16, yaxis_title_font_size=16,
                                  xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                  bargap=1.0) # Apply bargap here
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_placeholder("No 'AgentNextAction' data (excluding Unknown) for Sales calls.")
        else:
            display_placeholder("No 'AgentNextAction' data (excluding Unknown) for Sales calls.")
else:
        display_placeholder("Column 'AgentNextAction' not found or no Sales calls.")

#with col2:
display_business_question("Q7. How have Agents performed in Sales Calls?")
if col_agent_rating and not sales_df.empty:
        rating_counts = sales_df[col_agent_rating].value_counts()
        # Filter out 'Unknown'
        rating_counts = rating_counts[rating_counts.index != 'Unknown']
        
        total_ratings_q7 = rating_counts.sum()
        if total_ratings_q7 > 0:
            rating_order = ['High', 'Medium', 'Low'] # Explicit order for plotting, ensures consistency
            rating_counts = rating_counts.reindex(rating_order, fill_value=0).reset_index()
            rating_counts.columns = ['Rating', 'Count']
            # Filter out ratings with 0 count after reindexing if they were not present originally
            rating_counts = rating_counts[rating_counts['Count'] > 0] 

            rating_counts['Percentage'] = (rating_counts['Count'] / total_ratings_q7 * 100).round(1)
            rating_counts['text_label'] = rating_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

            if not rating_counts.empty:
                fig = px.bar(rating_counts, x='Rating', y='Count', text='text_label', color='Rating',
                             color_discrete_map=AGENT_RATING_COLOR_MAP, title="Agent Performance in Sales")
                fig.update_traces(textposition='outside', cliponaxis=False, width=0.5, textfont_size=16)
                fig.update_layout(height=400, showlegend=False, yaxis_range=[0, rating_counts['Count'].max() * 1.25],
                                  xaxis_title_font_size=16, yaxis_title_font_size=16,
                                  xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                  bargap=1.0) # Apply bargap here
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_placeholder("No 'AgentRating' data (excluding Unknown) for Sales calls.")
        else:
            display_placeholder("No 'AgentRating' data (excluding Unknown) for Sales calls.")
else:
        display_placeholder("Column 'AgentRating' not found or no Sales calls.")

# Add a separator
st.markdown("---")

# --- Service Insights ---
st.header("Service Insights")

display_business_question("Q2.What is the sentiment of Service Calls?")
if col_sentiment and not service_df.empty:
    sentiment_counts = service_df[col_sentiment].value_counts()
    # Filter out 'Unknown' sentiment
    sentiment_counts = sentiment_counts[sentiment_counts.index != 'Unknown']
    
    total_sentiment_service = sentiment_counts.sum()
    if total_sentiment_service > 0:
        sentiment_display_order = ['Positive', 'Neutral', 'Negative'] # Consistent order
        sentiment_counts = sentiment_counts.reindex(sentiment_display_order, fill_value=0).reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        # Filter out sentiments with 0 count after reindexing
        sentiment_counts = sentiment_counts[sentiment_counts['Count'] > 0]

        sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total_sentiment_service * 100).round(1)
        sentiment_counts['text_label'] = sentiment_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

        if not sentiment_counts.empty:
            fig = px.bar(sentiment_counts, x='Sentiment', y='Count', text='text_label', color='Sentiment',
                         color_discrete_map=SENTIMENT_COLOR_MAP, category_orders={"Sentiment": sentiment_display_order})
            fig.update_traces(textposition='outside', cliponaxis=False, width=0.5, textfont_size=16)
            fig.update_layout(height=400, title="Sentiment Distribution in Service Calls", showlegend=False,
                              yaxis_range=[0, sentiment_counts['Count'].max() * 1.25],
                              xaxis_title_font_size=16, yaxis_title_font_size=16,
                              xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                              bargap=1.0) # Apply bargap here
            st.plotly_chart(fig, use_container_width=True)
        else:
            display_placeholder("No sentiment data available for Service calls.")
    else:
        display_placeholder("No sentiment data available for Service calls.")
else:
    display_placeholder("Column 'CallSentiment' not found or no Service calls.")

display_business_question("Q3. What is the sentiment of Service Calls, broken down by Intent?")
if col_intent and col_sentiment and not service_df.empty:
    # Get raw counts first
    raw_counts_cross_tab = pd.crosstab(index=service_df[col_intent], columns=service_df[col_sentiment])
    
    # Get percentages (normalize by index means row-wise percentage)
    pct_cross_tab = pd.crosstab(index=service_df[col_intent], columns=service_df[col_sentiment], normalize='index').mul(100).round(1)
    
    sentiment_display_order = ['Positive', 'Neutral', 'Negative']
    intent_order_for_plot = ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
    
    # Reindex both tables to ensure consistent columns and row order
    raw_counts_cross_tab = raw_counts_cross_tab.reindex(index=[i for i in intent_order_for_plot if i in raw_counts_cross_tab.index], 
                                                        columns=sentiment_display_order, fill_value=0)
    pct_cross_tab = pct_cross_tab.reindex(index=[i for i in intent_order_for_plot if i in pct_cross_tab.index],
                                          columns=sentiment_display_order, fill_value=0)

    # Melt both into long format
    plot_df_counts = raw_counts_cross_tab.reset_index().melt(id_vars=col_intent, var_name='Sentiment', value_name='Count')
    plot_df_pct = pct_cross_tab.reset_index().melt(id_vars=col_intent, var_name='Sentiment', value_name='Percentage')

    # Merge to combine counts and percentages
    plot_df = pd.merge(plot_df_counts, plot_df_pct, on=[col_intent, 'Sentiment'])
    
    # Create the combined text label
    plot_df['text_label'] = plot_df.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)" if row['Count'] > 0 else "", axis=1)

    # Filter out rows where Percentage is 0 if text is empty (to avoid clutter if there's no data for that segment)
    plot_df = plot_df[plot_df['Count'] > 0] # Only plot segments that actually have calls

    fig = px.bar(plot_df, x=col_intent, y='Percentage', color='Sentiment', barmode='group',
                 text='text_label', # Use the new combined label
                 color_discrete_map=SENTIMENT_COLOR_MAP, 
                 category_orders={col_intent: intent_order_for_plot, "Sentiment": sentiment_display_order},
                 labels={'Percentage': 'Percentage of Calls', col_intent: 'User Intent to Buy', 'Count': 'Number of Calls'})
    fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=14) 
    fig.update_layout(height=500, yaxis_title='Sentiment Distribution (%)', legend_title_text='Sentiment', yaxis_range=[0, 115],
                      xaxis_title_font_size=16, yaxis_title_font_size=16,
                      xaxis_tickfont_size=14, yaxis_tickfont_size=14
                      ) 
    st.plotly_chart(fig, use_container_width=True)
else:
    display_placeholder("Cannot generate Intent vs. Sentiment chart. Check for 'UserIntentToBuy', 'CallSentiment', and Service calls.")

display_business_question("Q4.What are the Top Issues in Service Calls?")
if col_barrier and not service_df.empty:
    barrier_counts = service_df[col_barrier].value_counts().reset_index()
    barrier_counts.columns = ['Issue', 'Count']
    # Filter out non-descriptive issues
    barrier_counts = barrier_counts[~barrier_counts['Issue'].isin(['Not Specified', 'No Barrier / Info Unavailable'])]
    
    total_service_barriers = barrier_counts['Count'].sum()
    if total_service_barriers > 0:
        barrier_counts['Percentage'] = (barrier_counts['Count'] / total_service_barriers * 100).round(1)
        barrier_counts['text_label'] = barrier_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

        if not barrier_counts.empty:
            fig = px.bar(barrier_counts.head(10).sort_values('Count', ascending=True),
                         x='Count', y='Issue', orientation='h', text='text_label', title="Top Issues Raised in Service Calls",
                         color_discrete_sequence=[CUSTOM_COLORS['Biscay']]) # Use Biscay for barriers
            fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
            fig.update_layout(height=400, xaxis_title='Number of Mentions', yaxis_title='', xaxis_range=[0, barrier_counts['Count'].max() * 1.25],
                              xaxis_title_font_size=16, yaxis_title_font_size=16,
                              xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                              yaxis_automargin=True
                              ) 
            st.plotly_chart(fig, use_container_width=True)
        else:
            display_placeholder("No significant issue data found for Service calls.")
    else:
        display_placeholder("No significant issue data found for Service calls.")
else:
    display_placeholder("Column 'Major Purchase Barrier Theme' not found or no Service calls.")

display_business_question("Q5.What are the Talking Points in a Service Call?")
if col_top3themes and not service_df.empty:
    themes = service_df[col_top3themes].str.split(',').explode().str.strip()
    # Filter out non-descriptive themes
    themes = themes.dropna()[~themes.isin(['Not Specified', 'No Barrier / Info Unavailable', 'Other'])]
    
    total_service_themes = themes.shape[0] # Total mentions
    if total_service_themes > 0:
        theme_counts = themes.value_counts().reset_index()
        theme_counts.columns = ['Theme', 'Count']
        theme_counts['Percentage'] = (theme_counts['Count'] / total_service_themes * 100).round(1)
        theme_counts['text_label'] = theme_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

        if not theme_counts.empty:
            fig = px.bar(theme_counts.head(10).sort_values('Count', ascending=True),
                         x='Count', y='Theme', orientation='h', text='text_label', title="Top 10 Talking Points in Service Calls",
                         color_discrete_sequence=[CUSTOM_COLORS['Neptune']]) # Use Neptune for service themes
            fig.update_traces(textposition='outside', cliponaxis=False, textfont_size=16)
            fig.update_layout(height=450, xaxis_title='Number of Mentions', yaxis_title='', xaxis_range=[0, theme_counts['Count'].max() * 1.25],
                              xaxis_title_font_size=16, yaxis_title_font_size=16,
                              xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                              yaxis_automargin=True,
                              ) 
            st.plotly_chart(fig, use_container_width=True)
        else:
            display_placeholder("No talking points data available for Service calls.")
    else:
        display_placeholder("No talking points data available for Service calls.")
else:
    display_placeholder("Column 'Top3Themes' not found or no Service calls.")

# Service - Q6 and Q7 in columns
col3, col4 = st.columns(2)
#with col3:
display_business_question("Q6.How many Service Calls Require a Follow-up?")
if col_nextaction and not service_df.empty:
        action_counts = service_df[col_nextaction].value_counts()
        # Filter out 'Unknown'
        action_counts = action_counts[action_counts.index != 'Unknown']
        
        total_service_actions = action_counts.sum()
        if total_service_actions > 0:
            action_counts = action_counts.reset_index()
            action_counts.columns = ['Follow-up', 'Count']
            action_counts['Percentage'] = (action_counts['Count'] / total_service_actions * 100).round(1)
            action_counts['text_label'] = action_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

            if not action_counts.empty:
                fig = px.bar(action_counts, x='Follow-up', y='Count', text='text_label', color='Follow-up',
                             color_discrete_map={'Yes': CUSTOM_COLORS['Neptune'], 'No': CUSTOM_COLORS['Biscay']})
                fig.update_traces(textposition='outside', cliponaxis=False, width=0.4, textfont_size=16)
                fig.update_layout(height=400, title="Agent Follow-up for Service", showlegend=False, yaxis_range=[0, action_counts['Count'].max() * 1.25],
                                  xaxis_title_font_size=16, yaxis_title_font_size=16,
                                  xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                  bargap=1.0) # Apply bargap here
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_placeholder("No 'AgentNextAction' data (excluding Unknown) for Service calls.")
        else:
            display_placeholder("No 'AgentNextAction' data (excluding Unknown) for Service calls.")
else:
        display_placeholder("Column 'AgentNextAction' not found or no Service calls.")
#with col4:
display_business_question("How have Agents performed in Service Calls?")
if col_agent_rating and not service_df.empty:
        rating_counts = service_df[col_agent_rating].value_counts()
        # Filter out 'Unknown'
        rating_counts = rating_counts[rating_counts.index != 'Unknown']
        
        total_service_ratings = rating_counts.sum()
        if total_service_ratings > 0:
            rating_order = ['High', 'Medium', 'Low']
            rating_counts = rating_counts.reindex(rating_order, fill_value=0).reset_index()
            rating_counts.columns = ['Rating', 'Count']
            rating_counts = rating_counts[rating_counts['Count'] > 0]
            
            rating_counts['Percentage'] = (rating_counts['Count'] / total_service_ratings * 100).round(1)
            rating_counts['text_label'] = rating_counts.apply(lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1)

            if not rating_counts.empty:
                fig = px.bar(rating_counts, x='Rating', y='Count', text='text_label', color='Rating',
                             color_discrete_map=AGENT_RATING_COLOR_MAP, title="Agent Performance in Service")
                fig.update_traces(textposition='outside', cliponaxis=False, width=0.5, textfont_size=16)
                fig.update_layout(height=400, showlegend=False, yaxis_range=[0, rating_counts['Count'].max() * 1.25],
                                  xaxis_title_font_size=16, yaxis_title_font_size=16,
                                  xaxis_tickfont_size=14, yaxis_tickfont_size=14,
                                  bargap=1.0) # Apply bargap here
                st.plotly_chart(fig, use_container_width=True)
            else:
                display_placeholder("No 'AgentRating' data (excluding Unknown) for Service calls.")
        else:
            display_placeholder("No 'AgentRating' data (excluding Unknown) for Service calls.")
else:
        display_placeholder("Column 'AgentRating' not found or no Service calls.")

# Add a separator
st.markdown("---")

# --- Raw Data Expander ---
with st.expander("Show Filtered Raw Data"):
    st.dataframe(filtered_df)