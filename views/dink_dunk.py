import streamlit as st
from pathlib import Path
import base64
import pdfkit
import pandas as pd
import matplotlib.pyplot as plt
from joypy import joyplot
import matplotlib
import io

border_color = "rgb(40,40,40)"
shadow_color = "rgba(0, 0, 0, 0.1)"
hover_shadow_color = "rgba(0, 0, 0, 0.2)"

def categorize_field(yardline):
    if 0 <= yardline <= 25:
        return 'Opponent Red Zone'
    elif 26 <= yardline <= 50:
        return 'Midfield'
    elif 51 <= yardline <= 75:
        return 'Own Territory'
    elif 76 <= yardline <= 100:
        return 'Own Deep Zone'
    else:
        return 'Unknown'

@st.cache_data  
def _load_offense_tendency_data(offense_player_path, team_name):
    """Load offense data with caching."""
    try: 
        combo_path = offense_player_path + f"/{team_name}/full_route_combos.csv" 
        combo_df = pd.read_csv(combo_path)
        combo_df['field_zone'] = combo_df['yardline_100'].apply(categorize_field)
        
        use_cols = [
            "old_game_id", "play_id","series_result","series","series_success",
            "passer_player_name","passing_yards","receiver_player_name",
            "posteam","epa","yards_gained"
        ]
        pbp = pd.read_csv("assets/data/pbp_2022.csv", usecols=use_cols)
        pbp_df = pd.merge(
            combo_df, pbp, 
            left_on=['gameId', 'playId'], 
            right_on=['old_game_id', 'play_id'], 
            how='left'
        )
        return pbp_df
    except Exception as e:
        st.error(f"Error loading player data: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------
# CSS for styling the cards
# ----------------------------------------------------------------
st.markdown(f"""
<style>
.card {{
    border: 1px solid {border_color};
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    min-height: 150px;
    
}}

.row {{
    display: flex;
    flex-wrap: wrap;
    margin-top: 10px; /* Add some breathing room between rows */
}}

.column-30 {{
    flex: 30%;
    justify-content: center;  /* horizontally center */
    align-items: center; 
    padding-top: 4rem;
}}

.column-70 {{
    flex: 70%;
    padding: 10px;
}}

.column-100 {{
    flex: 100%;
    padding: 1rem;
}}

.logo {{
    max-width: 100%;
    height: auto;
}}

.left .order-1 {{
    order: 1;
}}

.left .order-2 {{
    order: 2;
}}

.right .order-1 {{
    order: 2;
}}

.right .order-2 {{
    order: 1;
}}
</style>
""", unsafe_allow_html=True)


st.title("The :blue[Dink] and :orange[Dunk] Report")

# -----------------------------------------------------------------------------
# Convert images to base64
# -----------------------------------------------------------------------------
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# -----------------------------------------------------------------------------
# Build advanced stats + ridge plot
# -----------------------------------------------------------------------------
def analyze_team_strategy(require_df, team_name, specific_strategies=None, colormap_name='tab20'):
    """Calculate the full set of stats for `team_name`, build an HTML bullet list, and create a ridge plot."""
    if specific_strategies is None:
        specific_strategies = ['Short Right Pass', 'Short Left Pass', 'Short Middle Pass']
    
    colormap = matplotlib.colormaps[colormap_name]
    
    # Filter the data for the team
    team_df = require_df[require_df['posteam'] == team_name]
    if team_df.empty:
        # Return a simple fallback if there's no data
        return f"<ul><li>No data available for {team_name}</li></ul>", None

    # Example computations (replace with your logic):
    # ----------------------------------------------------------------
    pass_length_percentages = require_df.groupby('posteam')['pass_length'].value_counts(normalize=True).unstack().fillna(0) * 100
    pass_length_percentages = pass_length_percentages.round(2)
    pass_length_counts = require_df.groupby('posteam')['pass_length'].value_counts().unstack().fillna(0)
    total_plays_series = pass_length_counts.sum(axis=1)

    success_counts = require_df[require_df['passResult'] == 'C'].groupby(['posteam','pass_length']).size().unstack(fill_value=0)
    success_rates = (success_counts / pass_length_counts * 100).fillna(0).round(2)
    overall_success_counts = require_df[require_df['passResult'] == 'C'].groupby('posteam').size()
    overall_success_rates = (overall_success_counts / total_plays_series * 100).round(2)
    
    strategy_counts = require_df.groupby(['posteam','strategy']).size().unstack(fill_value=0)
    strategy_success_counts = require_df[require_df['passResult'] == 'C'].groupby(['posteam','strategy']).size().unstack(fill_value=0)
    strategy_success_rates = (strategy_success_counts / strategy_counts * 100).fillna(0).round(2)

    team_total_plays = total_plays_series[team_name]
    short_count = pass_length_counts.loc[team_name, 'short']
    deep_count  = pass_length_counts.loc[team_name, 'deep']
    short_percent = pass_length_percentages.loc[team_name, 'short']
    deep_percent  = pass_length_percentages.loc[team_name, 'deep']
    short_success_rate = success_rates.loc[team_name, 'short']
    deep_success_rate  = success_rates.loc[team_name, 'deep']
    overall_success_rate = overall_success_rates[team_name]

    # First downs for short vs deep
    short_first_down_count = require_df[
        (require_df['posteam']==team_name) & 
        (require_df['pass_length']=='short') & 
        (require_df['series_result']=='First down')
    ].shape[0]
    deep_first_down_count = require_df[
        (require_df['posteam']==team_name) & 
        (require_df['pass_length']=='deep') & 
        (require_df['series_result']=='First down')
    ].shape[0]
    short_first_down_rate = (short_first_down_count / short_count * 100) if short_count else 0
    deep_first_down_rate  = (deep_first_down_count / deep_count * 100) if deep_count else 0
    
    # Yards and EPA
    short_yards_gained = require_df[
        (require_df['posteam']==team_name) & 
        (require_df['pass_length']=='short')
    ]['yards_gained'].sum()
    deep_yards_gained  = require_df[
        (require_df['posteam']==team_name) & 
        (require_df['pass_length']=='deep')
    ]['yards_gained'].sum()
    
    short_yards_per_play = (short_yards_gained / short_count) if short_count else 0
    deep_yards_per_play  = (deep_yards_gained / deep_count) if deep_count else 0
    
    short_epa_per_play = require_df[
        (require_df['posteam']==team_name) & 
        (require_df['pass_length']=='short')
    ]['epa'].mean()
    deep_epa_per_play  = require_df[
        (require_df['posteam']==team_name) & 
        (require_df['pass_length']=='deep')
    ]['epa'].mean()

    # Overall
    overall_first_down_count = require_df[
        (require_df['posteam']==team_name) & 
        (require_df['series_result']=='First down')
    ].shape[0]
    overall_first_down_rate = (overall_first_down_count / team_total_plays * 100) if team_total_plays else 0

    overall_yards_gained = require_df[
        (require_df['posteam']==team_name)
    ]['yards_gained'].sum()
    overall_yards_per_play = (overall_yards_gained / team_total_plays) if team_total_plays else 0
    overall_epa_per_play = require_df[
        (require_df['posteam']==team_name)
    ]['epa'].mean()

    # Build an HTML list
    summary_html = f"""
    <ul>
      <li><b>Total Pass Plays</b>: <i>{int(team_total_plays)}</i></li>
      <li><b>Short</b>: <i>{int(short_count)}</i> plays (<i>{short_percent:.2f}%</i>), <b>Success Rate</b>: <i>{short_success_rate:.2f}%</i>, 
          <b>First Down Rate</b>: <i>{short_first_down_rate:.2f}%</i>, 
          <b>Yards Per Play</b>: <i>{short_yards_per_play:.2f} yds</i>, 
          <b>EPA Per Play</b>: <i>{short_epa_per_play:.2f}</i></li>
      <li><b>Deep</b>: <i>{int(deep_count)}</i> plays (<i>{deep_percent:.2f}%</i>), <b>Success Rate</b>: <i>{deep_success_rate:.2f}%</i>, 
          <b>First Down Rate</b>: <i>{deep_first_down_rate:.2f}%</i>, 
          <b>Yards Per Play</b>: <i>{deep_yards_per_play:.2f} yds</i>, 
          <b>EPA Per Play</b>: <i>{deep_epa_per_play:.2f}</i></li>
      <li><b>Overall Success Rate</b>: <i>{overall_success_rate:.2f}%</i>, 
          <b>First Down Rate</b>: <i>{overall_first_down_rate:.2f}%</i>, 
          <b>Yards Per Play</b>: <i>{overall_yards_per_play:.2f} yds</i>, 
          <b>EPA Per Play</b>: <i>{overall_epa_per_play:.2f}</li></i>
      <br/>
      <li><b>Dink & Dunk Strategy</b>:
    </ul>
    """

    # Add each strategy bullet
    other_percentage = 100.0
    ridge_df_team = team_df[team_df['strategy'].isin(specific_strategies)]
    for strat in specific_strategies:
        if strat in strategy_counts.columns:
            strat_plays = strategy_counts.loc[team_name, strat]
            strat_success_rate = strategy_success_rates.loc[team_name, strat]
            strat_percent = (strat_plays / team_total_plays * 100) if team_total_plays else 0
            
            # First down rate
            strat_fd_count = require_df[
                (require_df['posteam']==team_name) &
                (require_df['strategy']==strat) &
                (require_df['series_result']=='First down')
            ].shape[0]
            strat_fd_rate = (strat_fd_count / strat_plays * 100) if strat_plays else 0
            
            # Yds & EPA
            yards_for_strat = require_df[
                (require_df['posteam']==team_name) &
                (require_df['strategy']==strat)
            ]['yards_gained'].sum()
            ypp_for_strat = (yards_for_strat / strat_plays) if strat_plays else 0
            epa_for_strat = require_df[
                (require_df['posteam']==team_name) &
                (require_df['strategy']==strat)
            ]['epa'].mean()
            
            other_percentage -= strat_percent
            
            summary_html += f"""
            <li><b>{strat}</b>: <i>{int(strat_plays)}</i> plays (<i>{strat_percent:.2f}%</i>), 
                <b>Success Rate</b>: <i>{strat_success_rate:.2f}%</i>, 
                <b>First Down Rate</b>: <i>{strat_fd_rate:.2f}%</i>, 
                <b>Yards Per Play</b>: <i>{ypp_for_strat:.2f} yds</i>, 
                <b>EPA Per Play</b>: <i>{epa_for_strat:.2f}</i></li> 
            """

    # Close the sub-list for Dink & Dunk
    summary_html += f"""
          <li><b>Other</b>: {other_percentage:.2f}%</li>
        </ul>
      </li>
    </ul>
    """

    # Build ridge plot with median lines
    _, ax = joyplot(
        ridge_df_team,
        by='strategy',
        column='epa',
        colormap=colormap, 
        labels=specific_strategies,
        linewidth=1.5,
        figsize=(8, 4),
        fade=True
    )

    medians = ridge_df_team.groupby('strategy')['epa'].median()
    means   = ridge_df_team.groupby('strategy')['epa'].mean()
    if not medians.empty:
        best_median_strat = medians.idxmax()
        best_median_value = medians[best_median_strat]
        for i, strategy in enumerate(specific_strategies):
            if strategy in medians.index:
                median_value = medians[strategy]
                mean_value   = means[strategy]
                ax[i].axvline(median_value, color='#F2EDD7FF', linestyle='-.', linewidth=1.5, label='Median')
                ax[i].axvline(mean_value,   color='black',     linestyle='--', linewidth=1, label='Mean')

    plt.xlabel("Expected Points Added (EPA)")
    plt.ylabel("Pass Strategies")
    plt.title(f"The strategy with the highest median EPA is: {best_median_strat} ({best_median_value:.2f})")

    return summary_html, plt.gcf()

# -----------------------------------------------------------------------------
# PDF creation
# -----------------------------------------------------------------------------
def download_team_pdf(team_name, html_summary, logo_base64, fig=None):
    """Build a PDF with the HTML bullet summary + chart."""
    chart_html = ""
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        chart_html = f'<img src="data:image/png;base64,{chart_base64}" style="max-width:600px;" />'
    
    full_html = f"""
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 1cm; }}
            body {{ font-family: Arial, sans-serif; font-size: 14px; line-height: 1.4; }}
            .card {{
                border: 2px solid black; 
                border-radius: 5px; 
                padding: 10px; 
                margin-bottom: 20px; 
                page-break-inside: avoid;
            }}
            .logo-container {{
                width: 30%; 
                float: left; 
                margin-right: 10px;
            }}
            .logo {{
                max-width: 100%; 
                max-height: 150px; 
                object-fit: contain;
            }}
            .description {{
                width: 65%; 
                float: right;
            }}
            .clearfix::after {{
                content: ""; 
                clear: both; 
                display: table;
            }}
            ul {{
                margin: 0; 
                padding-left: 1em;
            }}
            li {{
                margin-bottom: 4px;
            }}
            p {{
                margin-top: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="card clearfix">
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_base64}" class="logo">
            </div>
            <div class="description">
                {html_summary}
                {chart_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate PDF as a bytes object (in-memory)
    pdf_data = pdfkit.from_string(full_html, False)
    return pdf_data

def download_all_selected_pdf(content):
    full_html = f"""
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 1cm; }}
            body {{ font-family: Arial, sans-serif; }}
            .card {{
                border: 2px solid black; 
                border-radius: 5px; 
                padding: 10px; 
                margin-bottom: 20px; 
                page-break-inside: avoid;
            }}
            ul {{
                margin: 0; 
                padding-left: 1em;
            }}
            li {{
                margin-bottom: 4px;
            }}
            p {{
                margin-top: 8px;
            }}
        </style>
    </head>
    <body>
        {content}
    </body>
    </html>
    """

    pdf_data = pdfkit.from_string(full_html, False)
    return pdf_data

# -----------------------------------------------------------------------------
# Card display
# -----------------------------------------------------------------------------
def display_offensive_card(require_df, logo_folder, team_name, counter):
    layout_class = "left" if counter % 2 == 0 else "right"
    logo_file = logo_folder / f"{team_name}.png"
    logo_base64 = img_to_base64(str(logo_file)) if logo_file.exists() else ""

    # Analyze
    html_summary, fig = analyze_team_strategy(require_df, team_name)

    # Convert fig to base64 for inline display in the card
    chart_html = ""
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        chart_html = f'<img src="data:image/png;base64,{chart_base64}" style="width:100%; height:auto;" />'

    # Build final card HTML
    card_html = f"""
    <div class="card clearfix {layout_class}">
        <div class="row">
            <div class="column-30 {'order-1' if layout_class == 'left' else 'order-2'}">
                <img src="data:image/png;base64,{logo_base64}" class="logo">
            </div>
            <div class="column-70 {'order-2' if layout_class == 'left' else 'order-1'}">
                {chart_html}
            </div>
        </div>
        <div class="row">
            <div class="column-100">
                {html_summary}
            </div>
        </div>
    </div>
    """

    # Show in Streamlit
    st.html(card_html)

    # Create PDF
    team_pdf = download_team_pdf(team_name, html_summary, logo_base64, fig)
    
    # 2 columns for the download button
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.markdown(f'<div style="margin-top: 8px;">Download the <b>{team_name}</b> PDF:</div>', unsafe_allow_html=True)
    with col2:
        st.download_button(
            label="Download",
            data=team_pdf,
            file_name=f"{team_name}_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # Also return snippet for "Download All" PDF
    snippet_html = f"""
    <div class="card clearfix {layout_class}">
        <div class="row">
            <div class="column-70">
                {chart_html}
            </div>
            <div class="column-30">
                <img src="data:image/png;base64,{logo_base64}" class="logo">
            </div>
        </div>
        <div class="row">
            <div class="column-100">
                {html_summary}
            </div>
        </div>
    </div>
    """
    return snippet_html


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
offense_player_path = "assets/data/offense-data"
logo_folder = Path("assets/logo")
team_names = [f.stem for f in logo_folder.glob("*.png")]

selected_teams = st.multiselect(
    "Select Teams:", team_names, placeholder="Choose Teams", default=["PHI"]
)

all_teams_html = ""

if selected_teams:
    # Build a combined DataFrame or load individually
    require_df = pd.DataFrame()
    for team in selected_teams:
        temp_df = _load_offense_tendency_data(offense_player_path, team)
        require_df = pd.concat([require_df, temp_df], ignore_index=True)
    
    for idx, team in enumerate(selected_teams):
        snippet_html = display_offensive_card(require_df, logo_folder, team, idx)
        all_teams_html += snippet_html
    
    if all_teams_html.strip():
        combined_pdf = download_all_selected_pdf(all_teams_html)
        st.download_button(
            label="Download All Selected Report",
            data=combined_pdf,
            file_name="all_selected_offensive.pdf",
            mime="application/pdf",
            use_container_width=True
        )

