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

@st.cache_data(ttl=86400) 
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
    
                routes_str = " - ".join(route_list)
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
    if specific_strategies is None:
        specific_strategies = ['Short Right Pass', 'Short Left Pass', 'Short Middle Pass']
    
    colormap = plt.colormaps.get(colormap_name, plt.cm.tab20)
    team_df = require_df[require_df['posteam'] == team_name]
    if team_df.empty:
        return f"- No data available for {team_name}", None

    pass_length_counts = (require_df.groupby('posteam')['pass_length']
                          .value_counts()
                          .unstack()
                          .fillna(0))
    pass_length_percentages = (pass_length_counts
                               .div(pass_length_counts.sum(axis=1), axis=0) * 100
                               ).round(2)
    total_plays_series = pass_length_counts.sum(axis=1)

    success_counts = (require_df[require_df['passResult'] == 'C']
                      .groupby(['posteam','pass_length'])
                      .size()
                      .unstack(fill_value=0))
    success_rates = ((success_counts / pass_length_counts) * 100).fillna(0).round(2)

    overall_success_counts = (require_df[require_df['passResult'] == 'C']
                              .groupby('posteam')
                              .size())
    overall_success_rates = ((overall_success_counts / total_plays_series) * 100).round(2)
    
    strategy_counts = (require_df.groupby(['posteam','strategy'])
                       .size()
                       .unstack(fill_value=0))
    strategy_success_counts = (require_df[require_df['passResult'] == 'C']
                               .groupby(['posteam','strategy'])
                               .size()
                               .unstack(fill_value=0))
    strategy_success_rates = ((strategy_success_counts / strategy_counts) * 100).fillna(0).round(2)

    team_total_plays = total_plays_series.get(team_name, 0)
    short_count = pass_length_counts.loc[team_name, 'short'] if 'short' in pass_length_counts.columns else 0
    deep_count  = pass_length_counts.loc[team_name, 'deep']  if 'deep'  in pass_length_counts.columns else 0
    short_percent = pass_length_percentages.loc[team_name, 'short'] if 'short' in pass_length_percentages.columns else 0
    deep_percent  = pass_length_percentages.loc[team_name, 'deep']  if 'deep'  in pass_length_percentages.columns else 0
    short_success_rate = success_rates.loc[team_name, 'short'] if 'short' in success_rates.columns else 0
    deep_success_rate  = success_rates.loc[team_name, 'deep']  if 'deep'  in success_rates.columns else 0
    overall_success_rate = overall_success_rates.get(team_name, 0)

    short_first_down_count = team_df[
        (team_df['pass_length'] == 'short') &
        (team_df['series_result'] == 'First down')
    ].shape[0]
    deep_first_down_count = team_df[
        (team_df['pass_length'] == 'deep') &
        (team_df['series_result'] == 'First down')
    ].shape[0]
    short_first_down_rate = (short_first_down_count / short_count * 100) if short_count else 0
    deep_first_down_rate  = (deep_first_down_count / deep_count * 100) if deep_count else 0

    short_yards_gained = team_df[team_df['pass_length'] == 'short']['yards_gained'].sum()
    deep_yards_gained  = team_df[team_df['pass_length'] == 'deep']['yards_gained'].sum()
    short_yards_per_play = (short_yards_gained / short_count) if short_count else 0
    deep_yards_per_play  = (deep_yards_gained / deep_count) if deep_count else 0
    short_epa_per_play = team_df[team_df['pass_length'] == 'short']['epa'].mean()
    deep_epa_per_play  = team_df[team_df['pass_length'] == 'deep']['epa'].mean()

    overall_first_down_count = team_df[team_df['series_result'] == 'First down'].shape[0]
    overall_first_down_rate = (overall_first_down_count / team_total_plays * 100) if team_total_plays else 0

    overall_yards_gained = team_df['yards_gained'].sum()
    overall_yards_per_play = (overall_yards_gained / team_total_plays) if team_total_plays else 0
    overall_epa_per_play = team_df['epa'].mean()

    bullet_text_summary = (
        f"- **Total Pass Plays**: *{int(team_total_plays)}*\n"
        f"- **Short Pass**: *{int(short_count)}* plays (*{short_percent:.2f}%*), **Success Rate**: *{short_success_rate:.2f}%*, "
        f"**First Down Rate**: {short_first_down_rate:.2f}%, **Yards Per Play**: *{short_yards_per_play:.2f} yds*, "
        f"**EPA Per Play**: *{short_epa_per_play:.2f}*\n"
        f"- **Deep Pass**: *{int(deep_count)}* plays (*{deep_percent:.2f}%*), **Success Rate**: *{deep_success_rate:.2f}%*, "
        f"**First Down Rate**: *{deep_first_down_rate:.2f}%*, **Yards Per Play**: *{deep_yards_per_play:.2f} yds*, "
        f"**EPA Per Play**: *{deep_epa_per_play:.2f}*\n"
        f"- **Overall Success Rate**: *{overall_success_rate:.2f}%*, **First Down Rate**: *{overall_first_down_rate:.2f}%*, "
        f"**Yards Per Play**: *{overall_yards_per_play:.2f} yds*, **EPA Per Play**: *{overall_epa_per_play:.2f}*\n"
        f"- **Dink & Dunk Strategy**:"
    )

    other_percentage = 100.0
    ridge_df_team = team_df[team_df['strategy'].isin(specific_strategies)]
    for strat in specific_strategies:
        strat_plays = strategy_counts.loc[team_name, strat] if strat in strategy_counts.columns else 0
        strat_success_rate = strategy_success_rates.loc[team_name, strat] if strat in strategy_success_rates.columns else 0
        strat_percent = (strat_plays / team_total_plays * 100) if team_total_plays else 0

        strat_fd_count = team_df[
            (team_df['strategy'] == strat) &
            (team_df['series_result'] == 'First down')
        ].shape[0]
        strat_fd_rate = (strat_fd_count / strat_plays * 100) if strat_plays else 0

        yards_for_strat = team_df[team_df['strategy'] == strat]['yards_gained'].sum()
        ypp_for_strat = (yards_for_strat / strat_plays) if strat_plays else 0
        epa_for_strat = team_df[team_df['strategy'] == strat]['epa'].mean()

        other_percentage -= strat_percent

        bullet_text_summary += (
            f"\n  - **{strat}**: *{int(strat_plays)}* plays (*{strat_percent:.2f}%*), "
            f"**Success Rate**: *{strat_success_rate:.2f}%*, **First Down Rate**: *{strat_fd_rate:.2f}%*, "
            f"**Yards Per Play**: *{ypp_for_strat:.2f} yds*, **EPA Per Play**: *{epa_for_strat:.2f}*"
        )

    bullet_text_summary += f"\n  - **Other**: *{other_percentage:.2f}%*"

    if ridge_df_team.empty:
        return bullet_text_summary, None

    fig, axes = joyplot(
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
    means = ridge_df_team.groupby('strategy')['epa'].mean()


    if not medians.empty:
        best_median_strat = medians.idxmax()
        best_median_value = medians[best_median_strat]

        for i, strategy in enumerate(specific_strategies):
            if strategy in medians.index:
                median_value = medians[strategy]
                mean_value = means[strategy]
                axes[i].axvline(median_value, color='#F2EDD7', linestyle='-.', linewidth=1.5, label='Median')
                axes[i].axvline(mean_value, color='black', linestyle='--', linewidth=1, label='Mean')

        plt.title(f"The strategy with the highest median EPA is: {best_median_strat} ({best_median_value:.2f})")

    plt.xlabel("Expected Points Added (EPA)")
    plt.ylabel("Pass Strategies")

    return bullet_text_summary, fig


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
  

def format_text_to_html(input_text):
    bold_pattern = r'\*\*(.*?)\*\*'
    formatted_text = re.sub(bold_pattern, r'<b>\1</b>', input_text)
    italics_pattern = r'\*(.*?)\*'
    formatted_text = re.sub(italics_pattern, r'<i>\1</i>', formatted_text)
    
    return formatted_text

def display_team_info(routes_df, pass_receiver_df,bullet_text_summary, fig, team_name, logo_path=None):
    offense_table_html = show_offense_player_summary(routes_df, pass_receiver_df)
    logo_base64 = ""
    if logo_path and logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Add CSS styles for the containers
    container_style = """
        <style>
            .container1 {
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 20px;
            }
            .container2 {
                /* Add styles for Container 2 if needed */
            }
        </style>
    """

    # Display the CSS styles
    st.markdown(container_style, unsafe_allow_html=True)

    with st.container(border=True):
        col1, col2 = st.columns([3, 7])

        with col1:
            if logo_path and logo_path.exists():
                st.image(str(logo_path))
            else:
                st.write("No logo available.")

        with col2:
            if fig is not None:
                st.pyplot(fig)
            else:
                st.write("No figure generated.")
        
        
        st.markdown(bullet_text_summary)

        html_summary = f"<pre>{format_text_to_html(bullet_text_summary)}</pre>"
        team_pdf = download_team_pdf(
            team_name=team_name,
            html_summary=html_summary,
            logo_base64=logo_base64,
            offense_table_html=offense_table_html
        )

        st.download_button(
            label=f"Download {team_name} Dink & Dunk Report With Pass-Receiver Tendencies",
            data=team_pdf,
            file_name=f"{team_name}_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    st.divider()


@st.cache_data(ttl=86400)
def load_all_offense_data(offense_player_path):
    all_teams_data = {}
    for folder in Path(offense_player_path).iterdir():
        if folder.is_dir():
            team_name = folder.name
            pbp_df, routes_df, pass_receiver_df = _load_offense_tendency_data(offense_player_path, team_name)
            all_teams_data[team_name] = (pbp_df, routes_df, pass_receiver_df)
    return all_teams_data

all_data = load_all_offense_data("assets/data/offense-data")
logo_folder = Path("assets/logo")


if "team_order" not in st.session_state:
    st.session_state["team_order"] = {}


selected_teams = st.multiselect(
    "Select Teams:",
    list(all_data.keys()),
    placeholder="Choose Teams",
    default=["PHI"]
)

if selected_teams:
    for team in selected_teams:
        if team not in st.session_state["team_order"]:
            st.session_state["team_order"][team] = len(st.session_state["team_order"]) + 1

    selected_teams = sorted(
        selected_teams,
        key=lambda x: st.session_state["team_order"][x],
        reverse=True
    )

    require_df = pd.DataFrame()
    for team in selected_teams:
        pbp_df, routes_df, pass_receiver_df = all_data[team]
        require_df = pd.concat([require_df, pbp_df], ignore_index=True)
    
    for team in selected_teams:
        pbp_df, routes_df, pass_receiver_df = all_data[team]
        bullet_text_summary, fig = analyze_team_strategy(require_df, team)
        logo_path = logo_folder / f"{team}.png"

        display_team_info(
            routes_df=pass_receiver_df,
            pass_receiver_df=routes_df,
            bullet_text_summary=bullet_text_summary,
            fig=fig,
            team_name=team,
            logo_path=logo_path
        )

    

