import streamlit as st
from pathlib import Path
import base64
from xhtml2pdf import pisa
import pandas as pd
import matplotlib.pyplot as plt
from joypy import joyplot
import matplotlib
import io
import re

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
        routes_path = offense_player_path + f"/{team_name}/{team_name}_route_analysis.csv"
        pass_receiver_path = offense_player_path + f"/{team_name}/{team_name}_pass_receiver_analysis.csv"
        routes_df = pd.read_csv(routes_path)
        pass_receiver_df = pd.read_csv(pass_receiver_path)
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
        return pbp_df, routes_df, pass_receiver_df
    except Exception as e:
        st.error(f"Error loading player data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
def name_match(display_name, receiver_name):
    if pd.isna(display_name) or pd.isna(receiver_name):
        return False
    
    display_parts = display_name.split()
    receiver_parts = receiver_name.split('.')
    display_initial = display_parts[0][0].lower() 
    receiver_initial = receiver_parts[0][0].lower() 
    display_last = re.sub(r'[^a-zA-Z]', '', ''.join(display_parts[1:])).lower()
    receiver_last = re.sub(r'[^a-zA-Z]', '', ''.join(receiver_parts[1:])).lower()
    return display_initial == receiver_initial and display_last == receiver_last
    

def show_offense_player_summary(df_pass, df_route):
    summary = df_pass[[
        "passer_player_name", 
        "receiver_player_name", 
        "total_passes", 
        "C %", 
        "I %", 
        "C right %", 
        "C left %", 
        "C middle %"
    ]].copy()
    summary = summary[summary['total_passes'] > 1]
    summary['max_pct'] = summary[['C right %', 'C left %', 'C middle %']].max(axis=1)
    summary['preferred_side'] = summary[['C right %', 'C left %', 'C middle %']].idxmax(axis=1)\
                                    .str.replace('C ', '')\
                                    .str.replace(' %', '')\
                                    .str.capitalize()
    summary['preferred_location'] = summary['max_pct'].astype(str) + ' % on ' + summary['preferred_side']
    summary = summary.sort_values('max_pct', ascending=False)
    summary = summary.drop_duplicates(subset=['passer_player_name', 'receiver_player_name'], keep='first')
    summary = summary.drop(['max_pct', 'preferred_side', "C right %", "C left %", "C middle %"], axis=1)
    summary = summary.sort_values(by='total_passes', ascending=False)
    summary = summary.head(5)
    summary = summary.reset_index(drop=True)
    
    final_rows = []
    
    for _, row in summary.iterrows():
        passer = row['passer_player_name']
        receiver = row['receiver_player_name']
        total_passes = row['total_passes']
        c_percent = row['C %']
        i_percent = row['I %']
        preferred_location = row['preferred_location']
        matched_display_name = None
    
        for display_name in df_route['displayName'].unique():
            if name_match(display_name, receiver):
                matched_display_name = display_name
                break
    
        if matched_display_name:
            receiver_routes = df_route[df_route['displayName'] == matched_display_name]
            if not receiver_routes.empty:
                preferred_route = receiver_routes.loc[receiver_routes['C %'].idxmax()]
                routes_sorted = receiver_routes.sort_values(by='route %', ascending=False)
    
                top_routes = routes_sorted.head(4)
                other_routes = routes_sorted.iloc[4:]
    
                other_routes_sum = other_routes['route %'].sum()
                route_list = [f"{r['routeRan']}: {r['route %']}%" for _, r in top_routes.iterrows()]
                if other_routes_sum > 0:
                    route_list.append(f"Other: {other_routes_sum:.2f}%")
    
                routes_str = "\n - ".join(route_list)
            else:
                routes_str = "N/A"
                preferred_route = pd.Series({'routeRan': 'N/A', 'C %': 0})
        else:
            routes_str = "N/A"
            preferred_route = pd.Series({'routeRan': 'N/A', 'C %': 0})
    
        final_rows.append({
            'Passer - Receiver': f"{passer} - {receiver}",
            'Total Passes': total_passes,
            'Complete Pass': f"{round(c_percent, 2)} %",
            'Incomplete Pass': f"{round(i_percent, 2)} %",
            'Preferred Location': preferred_location,
            'Successful Route': f"{preferred_route['routeRan']} ({preferred_route['C %']:.2f}% Completion)",
            'Routes Ran %': routes_str,
        })
    
    final_df = pd.DataFrame(final_rows)
    offense_table_html = final_df.to_html(index=False, classes="offense-table", border=1)
    return offense_table_html


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


st.title("The :blue[Dink] & :orange[Dunk] Report")

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def analyze_team_strategy(require_df, team_name, specific_strategies=None, colormap_name='tab20'):
    """Calculate the full set of stats for `team_name`, build an HTML bullet list, and create a ridge plot."""
    if specific_strategies is None:
        specific_strategies = ['Short Right Pass', 'Short Left Pass', 'Short Middle Pass']
    
    colormap = matplotlib.colormaps[colormap_name]
    team_df = require_df[require_df['posteam'] == team_name]
    if team_df.empty:
        return f"<ul><li>No data available for {team_name}</li></ul>", None

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

    summary_html += f"""
          <li><b>Other</b>: {other_percentage:.2f}%</li>
        </ul>
      </li>
    </ul> 
    """

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


def html_to_pdf_bytes(html_str):
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html_str), dest=pdf_buffer)
    if pisa_status.err:
        return None
    return pdf_buffer.getvalue()

def download_team_pdf(team_name, html_summary, logo_base64, fig=None, offense_table_html=""):
    full_html = f"""
    <html>
    <head>
        <style>
            @page {{
                size: A4 landscape;
                margin: 0.5cm;
            }}
            body {{
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.2;
                margin: 0;
                padding: 0;
            }}

            .row {{
                display: flex;
                align-items: center; /* optional; keeps content aligned in the center vertically */
            }}

            .column {{
                flex: 1;            /* each .column takes equal horizontal space */
                margin-right: 10px; /* optional; spacing between columns */
            }}          
            
            .offense-table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 10px;
                table-layout: fixed;
            }}
            .offense-table th, .offense-table td {{
                border: 1px solid #ddd;
                padding: 4px;
                text-align: center;
            }}
            .offense-table th {{
                font-weight: bold;
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="row">
            <div class="column">
                <center><img src="data:image/png;base64,{logo_base64}" alt="Team Logo" width="100" height="100"></center>
            </div>
            <div class="column">
                {html_summary}
            </div>
        </div>
        <br/>
        <h3>Top 5 Pass-Receiver Tendencies</h3>
        {offense_table_html}
        
    </body>

    </html>
    """

    pdf_bytes = html_to_pdf_bytes(full_html)
    return pdf_bytes
  
 
def create_multi_team_pdf(team_snippets):
    full_html = f"""
    <html>
    <head>
        <style>
            @page {{
                size: A4 landscape;
                margin: 0.5cm;
            }}
            body {{
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.2;
                margin: 0;
                padding: 0;
            }}
            .row {{
                display: flex;
                align-items: center;
            }}
            .column {{
                flex: 1;
                margin-right: 10px;
            }}
            .offense-table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 10px;
                table-layout: fixed;
            }}
            .offense-table th, .offense-table td {{
                border: 1px solid #ddd;
                padding: 4px;
                text-align: center;
            }}
            .offense-table th {{
                font-weight: bold;
                background-color: #f2f2f2;
            }}
            .team-section {{
                page-break-after: always;
            }}
        </style>
    </head>
    <body>
        {team_snippets}
    </body>
    </html>
    """
    return html_to_pdf_bytes(full_html)


def display_offensive_card(require_df, routes_df, pass_receiver_df, logo_folder, team_name, counter):
    layout_class = "left" if counter % 2 == 0 else "right"
    logo_file = logo_folder / f"{team_name}.png"
    logo_base64 = img_to_base64(str(logo_file)) if logo_file.exists() else ""
    offense_table_html = show_offense_player_summary(routes_df, pass_receiver_df)
    html_summary, fig = analyze_team_strategy(require_df, team_name)
    chart_html = ""
    chart_base64 = ""
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        chart_html = f'<img src="data:image/png;base64,{chart_base64}" style="width:600px; height:auto;" />'

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
    st.html(card_html)
    team_pdf = download_team_pdf(
        team_name=team_name,
        html_summary=html_summary,
        logo_base64=logo_base64,
        fig=fig,
        offense_table_html=offense_table_html
    )
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
                <br>
                {offense_table_html}
            </div>
        </div>
    </div>
    """
    return snippet_html



offense_player_path = "assets/data/offense-data"
logo_folder = Path("assets/logo")
team_names = [f.stem for f in logo_folder.glob("*.png")]

selected_teams = st.multiselect(
    "Select Teams:", team_names, placeholder="Choose Teams", default=["PHI"]
)

all_teams_html = ""

if selected_teams:
    require_df = pd.DataFrame()
    for team in selected_teams:
        temp_df, routes_df, pass_receiver_df = _load_offense_tendency_data(offense_player_path, team)
        require_df = pd.concat([require_df, temp_df], ignore_index=True)
    
    for idx, team in enumerate(selected_teams):
        snippet_html = display_offensive_card(require_df, pass_receiver_df,routes_df, logo_folder, team, idx)
        all_teams_html += snippet_html
    

