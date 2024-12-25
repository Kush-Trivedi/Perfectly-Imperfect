import os
import re
import json
import base64
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


model_directory ="assets/models/yards_gained"
offense_model_dir = "assets/models/offense_strategy"
player_data = "assets/data/all_player_ratings.csv"
historical_data = "assets/data/season_2022_historical_data.csv"
logo_folder_path = "assets/logo"
offense_player_path = "assets/data/offense-data"

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
    epsilon = 1e-6
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

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

@st.cache_resource
def load_model_components(model_dir):
    try:
        model_file = os.path.join(model_dir, 'yard_gained_model.keras')
        model = tf.keras.models.load_model(model_file, compile=False, custom_objects={'focal_loss_fixed': focal_loss_fixed})
        cat_mapping_file = os.path.join(model_dir, 'cat_mapping.json')

        with open(cat_mapping_file, 'r') as f:
            cat_mapping = json.load(f)

        scaler_file = os.path.join(model_dir, 'scaler.joblib')
        scaler = joblib.load(scaler_file)

        y_labels_file = os.path.join(model_dir, 'y_labels.json')
        with open(y_labels_file, 'r') as f:
            y_labels = json.load(f)

        return model, cat_mapping, scaler, y_labels

    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None, None
    
@st.cache_resource
def load_off_startegy_model_components(offense_model_dir):
    try:
        model_file = os.path.join(offense_model_dir, 'offense_strategy_model.keras')
        model = tf.keras.models.load_model(model_file, compile=False, custom_objects={'focal_loss_fixed': focal_loss_fixed})

        cat_mapping_file = os.path.join(offense_model_dir, 'offense_strategy_cat_mapping.json')
        with open(cat_mapping_file, 'r') as f:
            cat_mapping = json.load(f)

        scaler_file = os.path.join(offense_model_dir, 'offense_strategy_scaler.joblib')
        scaler = joblib.load(scaler_file)

        y_labels_file = os.path.join(offense_model_dir, 'offense_strategy_y_labels.json')
        with open(y_labels_file, 'r') as f:
            y_labels = json.load(f)

        return model, cat_mapping, scaler, y_labels

    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None, None
    

model, cat_mapping, scaler, y_labels = load_model_components(model_directory)
offense_strategy_model, offense_strategy_cat_mapping, offense_strategy_scaler, offense_strategy_y_labels = load_off_startegy_model_components(offense_model_dir)
    

@st.cache_data
def _load_player_data(player_data_path):
    try:
        return pd.read_csv(player_data_path)
    except Exception as e:
        st.error(f"Error loading player data: {e}")
        return pd.DataFrame()
    
@st.cache_data  
def _load_offense_tendency_data(offense_player_path, team_name):
    try:
        routes_path = offense_player_path + f"/{team_name}/{team_name}_route_analysis.csv"
        pass_receiver_path = offense_player_path + f"/{team_name}/{team_name}_pass_receiver_analysis.csv"
        combo_path = offense_player_path + f"/{team_name}/full_route_combos.csv"  # Added missing '/'
        routes_df = pd.read_csv(routes_path)
        pass_receiver_df = pd.read_csv(pass_receiver_path)
        combo_df = pd.read_csv(combo_path)
        combo_df['field_zone'] = combo_df['yardline_100'].apply(categorize_field)  # Corrected to combo_df
        return routes_df, pass_receiver_df, combo_df
    except Exception as e:
        st.error(f"Error loading player data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    

@st.cache_data
def _load_historical_data(historical_data_path):
    try:
        return pd.read_csv(historical_data_path)
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.DataFrame()
    
@st.cache_data
def load_ratings(quarter, down):
    try:
        file_path = f"assets/data/all_{quarter}_{down}_player_rating.csv"
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found for Quarter {quarter} and Down {down}.")
        return pd.DataFrame()

@st.cache_data
def get_weighted_player_rating(player_df, player_names, quarter, down):
    overall_df = player_df
    specific_df = load_ratings(quarter, down)

    if specific_df.empty:
        return []

    overall_filtered = overall_df[overall_df['displayName'].isin(player_names)]
    specific_filtered = specific_df[specific_df['displayName'].isin(player_names)]
    merged_ratings = pd.merge(
        overall_filtered[['displayName', 'final_rating']],
        specific_filtered[['displayName', 'final_rating']],
        on='displayName',
        suffixes=('_overall', '_specific')
    )
    merged_ratings['final_weighted_rating'] = (
        0.3 * merged_ratings['final_rating_overall'] +
        0.7 * merged_ratings['final_rating_specific']
    )
    missing_players = set(player_names) - set(merged_ratings['displayName'])
    if missing_players:
        st.warning(f"Ratings not found for: {', '.join(missing_players)}")
    return merged_ratings['final_weighted_rating'].tolist()


def predict_yard_bins_sorted(game_scenario, model, cat_features, cat_mapping, numerical_features, scaler, y_labels):
    scenario_df = pd.DataFrame([game_scenario])

    for cat_col in cat_features:
        scenario_df[cat_col] = scenario_df[cat_col].astype(str).map(cat_mapping[cat_col])
    
    for cat_col in cat_features:
        if scenario_df[cat_col].isnull().any():
            unknown_index = cat_mapping[cat_col].get('unknown', 0)
            scenario_df[cat_col] = scenario_df[cat_col].fillna(unknown_index).astype(int)
    
    scenario_df[numerical_features] = scaler.transform(scenario_df[numerical_features])
    
    X_cat = scenario_df[cat_features].values 
    X_num = scenario_df[numerical_features].values

    X_cat_tensor = tf.convert_to_tensor(X_cat, dtype=tf.int32)
    X_num_tensor = tf.convert_to_tensor(X_num, dtype=tf.float32)

    y_pred = model([X_cat_tensor, X_num_tensor], training=False)
    probabilities = y_pred.numpy()[0]
    bin_probs = {y_labels[i]: float(prob) for i, prob in enumerate(probabilities)}
    sorted_bin_probs = dict(sorted(bin_probs.items(), key=lambda item: item[1], reverse=True))
    return sorted_bin_probs

def predict_strategy_sorted(game_scenario, model, cat_features, cat_mapping, numerical_features, scaler, y_labels):
    scenario_df = pd.DataFrame([game_scenario])

    for cat_col in cat_features:
        scenario_df[cat_col] = scenario_df[cat_col].astype(str).map(cat_mapping[cat_col])
    
    for cat_col in cat_features:
        if scenario_df[cat_col].isnull().any():
            unknown_index = cat_mapping[cat_col].get('unknown', 0)
            scenario_df[cat_col] = scenario_df[cat_col].fillna(unknown_index).astype(int)
    
    scenario_df[numerical_features] = scaler.transform(scenario_df[numerical_features])
    X_cat = scenario_df[cat_features].values 
    X_num = scenario_df[numerical_features].values

    X_cat_tensor = tf.convert_to_tensor(X_cat, dtype=tf.int32)
    X_num_tensor = tf.convert_to_tensor(X_num, dtype=tf.float32)

    y_pred = model([X_cat_tensor, X_num_tensor], training=False)
    probabilities = y_pred.numpy()[0]

    bin_probs = {y_labels[i]: float(prob) for i, prob in enumerate(probabilities)}
    sorted_probs = dict(sorted(bin_probs.items(), key=lambda item: item[1], reverse=True))
    return sorted_probs


class NFLAdvancedPlaygroundSimulator:
    def __init__(self, model_dir, player_data_path, historical_data_path, logo_folder_path,offense_player_path):
        self.model_dir = model_dir
        self.player_data_path = player_data_path
        self.historical_data_path = historical_data_path
        self.offense_player_path = offense_player_path
        self.loaded_model, self.loaded_encoder, self.loaded_scaler, self.loaded_y_labels = load_model_components(self.model_dir)
        self.player_df = _load_player_data(self.player_data_path)
        self.historical_data = _load_historical_data(self.historical_data_path)
        self._set_plot_defaults()
        self._initialize_categories()
        self.logo_folder = Path(logo_folder_path)
        self.team_names = self._load_team_names()
        self._team_full_names()
        self.routes_df = pd.DataFrame()
        self.pass_receiver_df = pd.DataFrame()
        self.combo_df = pd.DataFrame()


    @staticmethod
    def _set_plot_defaults():
        plt.rcParams['figure.dpi'] = 180
        plt.rcParams["figure.figsize"] = (5, 5)
        sns.set_theme(rc={
            'axes.facecolor': '#FFFFFF',
            'figure.facecolor': '#FFFFFF',
            'font.sans-serif': 'DejaVu Sans',
            'font.family': 'sans-serif'
        })

    def img_to_base64(self,image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
        
    def _team_full_names(self):
        self.team_full_names = {
            'LA': 'Los Angeles Rams',
            'ATL': 'Atlanta Falcons',
            'CAR': 'Carolina Panthers',
            'CHI': 'Chicago Bears',
            'CIN': 'Cincinnati Bengals',
            'DET': 'Detroit Lions',
            'HOU': 'Houston Texans',
            'MIA': 'Miami Dolphins',
            'NYJ': 'New York Jets',
            'WAS': 'Washington Commanders',
            'ARI': 'Arizona Cardinals',
            'LAC': 'Los Angeles Chargers',
            'MIN': 'Minnesota Vikings',
            'TEN': 'Tennessee Titans',
            'DAL': 'Dallas Cowboys',
            'SEA': 'Seattle Seahawks',
            'KC': 'Kansas City Chiefs',
            'BAL': 'Baltimore Ravens',
            'CLE': 'Cleveland Browns',
            'JAX': 'Jacksonville Jaguars',
            'NO': 'New Orleans Saints',
            'NYG': 'New York Giants',
            'PIT': 'Pittsburgh Steelers',
            'SF': 'San Francisco 49ers',
            'DEN': 'Denver Broncos',
            'LV': 'Las Vegas Raiders',
            'GB': 'Green Bay Packers',
            'BUF': 'Buffalo Bills',
            'PHI': 'Philadelphia Eagles',
            'IND': 'Indianapolis Colts',
            'NE': 'New England Patriots',
            'TB': 'Tampa Bay Buccaneers'
        }

    def _initialize_categories(self):
        self.offense_categories = {
            "blocking": ["Run Block", "Pass Block", "Lead Block", "Kneel Block", "Quick Block", "Spike Assist"],
            "routes": ["HITCH", "POST", "CORNER", "OUT", "GO", "SCREEN", "FLAT", "ANGLE", "CROSS", "IN", "SLANT", "WHEEL"],
            "special_actions": ["Kneel", "Kneel Assist", "Spike", "Stationary"],
            "no_assignment": ["No Route"]
        }
        self.defense_categories = {
            "run_defense": ["Contain", "Deep Run Support", "Run Contain", "Cutback Contain", "General Run Defense", "Force", "Gap Control", "Gap Fill", "Inside Gap"],
            "zone_defense": ["Zone Coverage", "Hook Zone", "Flat Zone Right", "Flat Zone Left", "Deep Zone", "Deep Third Right", "HCR", "HCL", "CFL"],
            "man_defense": ["MAN", "HOL", "FR", "3M", "3R", "3L", "2R", "2L", "FL"],
            "pass_rush": ["Pass Rush", "4IR", "4OL", "4IL", "4OR", "DF", "PRE"],
            "miscellaneous": ["Unknown Assignment", "CFR"],
            "special_defense": ["Deep Run Support", "Deep Third Right", "Flat Zone Right", "Flat Zone Left"]
        }
        self.offensive_position_categories = {
            "skill_players": ["QB", "WR", "TE", "RB", "FB"],
            "offensive_line": ["T", "G", "C"],
            "uncommon_positions": ["OLB", "ILB", "DT", "FS"]
        }
        self.defensive_position_categories = {
            "defensive_line": ["DT", "DE", "NT"],
            "linebackers": ["OLB", "MLB", "ILB", "LB"],
            "defensive_backs": ["FS", "SS", "CB", "DB"],
            "non_traditional": ["QB", "WR"]
        }
    
    def _load_team_names(self):
        if not self.logo_folder.exists():
            st.error(f"Logo folder not found: {self.logo_folder}")
            return []
        
        team_names = [logo_file.stem for logo_file in self.logo_folder.glob("*.png")]
        
        if "NE" in team_names:
            team_names.remove("NE")          
            team_names.insert(0, "NE")      

        if "IND" in team_names:
            team_names.remove("IND")        
            team_names.insert(1, "IND")  
        
        return team_names

    def get_player_rating(self, player_names):
        filtered_players = self.player_df[self.player_df['displayName'].isin(player_names)]
        ratings = filtered_players['final_rating'].tolist()
        missing_players = set(player_names) - set(filtered_players['displayName'])
        if missing_players:
            st.warning(f"Ratings not found for: {', '.join(missing_players)}")
        return ratings
    
    def calculate_off_def_proportions(self,selected_routes, categories):
        total_routes = len(selected_routes)
        if total_routes == 0:
            return {f"{category}_ratio": 0 for category in categories.keys()}
        
        ratios = {}
        for category, values in categories.items():
            count = sum(route in values for route in selected_routes)
            ratios[f"{category}_ratio"] = count / total_routes
        return ratios

    def calculate_position_features(self,selected_positions, categories):
        total_positions = len(selected_positions)
        if total_positions == 0:
            return {f"{category}_count": 0 for category in categories.keys()} | {f"{category}_ratio": 0 for category in categories.keys()}
        
        features = {}
        for category, positions in categories.items():
            count = sum(pos in positions for pos in selected_positions)
            features[f"{category}_count"] = count
            features[f"{category}_ratio"] = count / total_positions
        return features
    
    def name_match(self,display_name, receiver_name):
        if pd.isna(display_name) or pd.isna(receiver_name):
            return False
        
        display_parts = display_name.split()
        receiver_parts = receiver_name.split('.')
        display_initial = display_parts[0][0].lower() 
        receiver_initial = receiver_parts[0][0].lower() 
        display_last = re.sub(r'[^a-zA-Z]', '', ''.join(display_parts[1:])).lower()
        receiver_last = re.sub(r'[^a-zA-Z]', '', ''.join(receiver_parts[1:])).lower()
        return display_initial == receiver_initial and display_last == receiver_last



    def genrate_payload(self,
            quarter, down, yards_to_go, yards_to_endzone, game_half, is_second_half, 
            quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining,
            offense_play_type, off_score, off_timeout_remaining, 
            offense_wp, offense_formation, offense_selected_positions, 
            offense_selected_routes, offense_ratings,
            defense_play_type, def_score, def_timeout_remaining, defense_wp, 
            defense_formation, defense_selected_positions, 
            defense_selected_assignments, defense_ratings
        ):
        quarter_time_ratio = round(quarter_seconds_remaining / 900, 6)
        half_time_ratio = round(half_seconds_remaining / 1800, 6)
        normalized_down = round(down / 4, 6)
        normalized_quarter = round(quarter / 4, 6)
        situation_pressure = round(
            (normalized_down + normalized_quarter + quarter_time_ratio) / 3, 6
        )
        offense_defense_interaction = offense_formation + "_" + defense_formation

        posteam_timeout_usage = round((3 - off_timeout_remaining) / 3, 6)
        defteam_timeout_usage = round((3 - def_timeout_remaining) / 3, 6)

        offense_ratios = self.calculate_off_def_proportions(offense_selected_routes, self.offense_categories)
        defense_ratios = self.calculate_off_def_proportions(defense_selected_assignments, self.defense_categories)
        offense_position_features = self.calculate_position_features(offense_selected_positions, self.offensive_position_categories)
        defense_position_features = self.calculate_position_features(defense_selected_positions, self.defensive_position_categories)

        game_scenario = {
            "quarter": quarter,
            "down": down,
            "yardsToGo": yards_to_go,
            "absoluteYardlineNumber": yards_to_endzone, 
            "offenseFormation": offense_formation,
            "play_type": offense_play_type,
            "defenseFormation": defense_formation,
            "pff_manZone": defense_play_type,
            "average_offense_rating": offense_ratings,
            "average_defense_rating": defense_ratings,
            "quarter_seconds_remaining": quarter_seconds_remaining,
            "half_seconds_remaining": half_seconds_remaining,
            "game_seconds_remaining": game_seconds_remaining,
            "game_half": game_half,
            "goal_to_go": int(yards_to_go <= 10 and yards_to_endzone >= 90), 
            "score_differential": float(off_score-def_score),  
            "wp": offense_wp,
            "def_wp": defense_wp,
            "down_yardsToGo": down * yards_to_go,
            "quarter_time_ratio": quarter_time_ratio,
            "half_time_ratio": half_time_ratio,
            "posteam_timeout_usage": posteam_timeout_usage,
            "defteam_timeout_usage": defteam_timeout_usage,
            "is_second_half": is_second_half,
            "normalized_down": normalized_down,
            "normalized_quarter": normalized_quarter,
            "normalized_time_remaining": quarter_time_ratio,
            "situation_pressure": situation_pressure,
            "offense_defense_interaction": offense_defense_interaction,
            **offense_ratios,
            **defense_ratios,
            **offense_position_features,
            **defense_position_features
        }

        return game_scenario
    
    def game_situation_components(self):
        yards_to_go_options = list(range(1, 31))
        yards_to_endzone_options = list(range(1, 101))
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            quarter = st.selectbox("Quarter", options=[1, 2, 3, 4, 5],index=[1, 2, 3, 4].index(3) if 3 in [1, 2, 3, 4, 5] else 0, key="quarter")
            st.markdown(
                """
                    <style> div[data-baseweb="segmented-control"] { } div[data-baseweb="segmented-control"] button { } </style>
                """,
                unsafe_allow_html=True
            )
            option_map = {0: ":material/hourglass_top: ", 1: ":material/hourglass_bottom: ", 2: ":material/more_time: "}
            game_half = st.segmented_control( "Game Time Segment", options=option_map.keys(), format_func=lambda option: option_map[option], default=1)
            if game_half == 0:
                selected_half = f"{option_map[game_half]} Half - 1"
            elif game_half == 1:
                selected_half = f"{option_map[game_half]} Half - 2"
            elif game_half == 2:
                selected_half = f"{option_map[game_half]} Overtime"
            else:
                selected_half = "Unknown"

            st.markdown(f"Phase of Play: **{selected_half}**")
            selected_half = selected_half.split(":")[-1].strip()
            game_half = "Half1" if selected_half == "Half - 1" else ("Half2" if selected_half == "Half - 2" else "Overtime")
            is_second_half = 1 if game_half == "Half2" else 0
            
        with col2:    
            down = st.selectbox("Down", options=[1, 2, 3, 4], index=[0, 1, 2, 3].index(2) if 3 in [1, 2, 3, 4] else 0, key="down")
            quarter_seconds_remaining = st.slider("Seconds Remaining in a Quarter", min_value=0, max_value=900,step=1, value=268)
            minutes = quarter_seconds_remaining // 60
            seconds = quarter_seconds_remaining % 60
            st.markdown(f"Time left in Qtr: **{minutes:02}:{seconds:02}** (*MM:SS*)")
            
        with col3:
            yards_to_go = st.selectbox("Yards to Go", options=yards_to_go_options, index=4-1 if 4 in yards_to_go_options else 0, key="yards_to_go")
            half_seconds_remaining = st.slider("Seconds Remaining in a Half", min_value=0, max_value=1800, value=1168)
            minutes = half_seconds_remaining // 60
            seconds = half_seconds_remaining % 60
            st.markdown(f"Half time left: **{minutes:02}:{seconds:02}** (*MM:SS*)")

        with col4:
            yards_to_endzone = st.selectbox("Yards to Endzone", options=yards_to_endzone_options,index=85-1 if 85 in yards_to_endzone_options else 0, key="yards_endzone")
            game_seconds_remaining = st.slider("Seconds Remaining in the Game", min_value=0, max_value=3600, value=1168)
            minutes = game_seconds_remaining // 60
            seconds = game_seconds_remaining % 60
            st.markdown(f"Game time left: **{minutes:02}:{seconds:02}** (*MM:SS*)")

        st.divider()

        return quarter, down, yards_to_go, yards_to_endzone, game_half, is_second_half, quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining
    
    def offense_details(self, offense_col, quarter, down):
        offense_selected_players = []
        offense_selected_positions = []
        offense_selected_routes = []
        unique_offense_formation = ['SINGLEBACK', 'SHOTGUN', 'EMPTY', 'PISTOL','I_FORM', 'JUMBO', 'WILDCAT']
        unique_offense_position = ["QB", "WR", "TE", "RB", "FB", "T", "G", "C"]
        unique_routes =[
            "Run Block", "Pass Block", "Lead Block", "Kneel Block", "Quick Block", "Spike Assist", "HITCH", "POST", "CORNER", "OUT", 
            "GO", "SCREEN", "FLAT", "ANGLE", "CROSS", "IN", "SLANT", "WHEEL", "Kneel", "Kneel Assist", "Spike", "Stationary","No Route"
        ] 

        with offense_col:
            offense_col_a, offense_col_b = st.columns(2)
            with offense_col_a:
                st.subheader(":green[Offense]", divider="gray")

            with offense_col_b:
                offense_play_option_map = { 0: ":material/sprint: Run", 1: ":material/arrow_split: Pass" }
                offense_play_selection = st.pills("Play Type", options=offense_play_option_map.keys(), format_func=lambda option: offense_play_option_map[option], selection_mode="single", default=1, key="offense_play_selection")
                offense_play_type = offense_play_option_map[offense_play_selection]
                offense_play_type = offense_play_type.split(":")[-1].strip().lower()


            offense_col_c, offense_col_d, offense_col_e, offense_col_f = st.columns(4)

            with offense_col_c:
                offense_team_name = st.selectbox("Offense Team", options=self.team_names, key="offense_team_name")
                self.routes_df, self.pass_receiver_df, self.combo_df = _load_offense_tendency_data(self.offense_player_path, offense_team_name)

            with offense_col_d:
                off_score = st.number_input("Score Points", key="offense_score", min_value=0, max_value=99, step=1,value=13,format="%d")

            with offense_col_e:
                off_timeout_remaining = st.selectbox("Timeout Left", options=[3, 2, 1])

            with offense_col_f:
                offense_wp = st.number_input("Win Probability", value=0.90206, format="%.5f")

            selected_team_logo_path = self.logo_folder / f"{offense_team_name}.png"
            if selected_team_logo_path.exists():
                img_base64 = self.img_to_base64(str(selected_team_logo_path))
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
                        <img src="data:image/png;base64,{img_base64}" alt="{offense_team_name}" 
                            style="width: 250px; height: 250px; object-fit: contain;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"Logo not found for team: {offense_team_name}")

            offense_formation = st.selectbox("Offense Formation", options= unique_offense_formation, index=1)

            default_offense_players = [
                "James Ferentz", "Trenton Brown", "Hunter Henry", "Kendrick Bourne",
                "Isaiah Wynn", "Yodny Cajuste", "Jakobi Meyers", "Michael Onwenu",
                "Mac Jones", "Rhamondre Stevenson", "Tyquan Thornton"
            ]

            default_offense_positions = [
                "C", "T", "TE", "WR", "G",
                "T", "WR", "G", "QB", "RB", "WR"
            ]

            default_offense_routes = [
                "Pass Block", "Pass Block", "CORNER", "HITCH", "Pass Block",
                "Pass Block", "IN", "Pass Block", "No Route", "OUT", "GO"
            ]

            for i in range(11):
                col_player, col_position, col_route = st.columns(3)
                
                with col_player:
                    player_options = self.player_df['displayName'].tolist()
                    
                    if default_offense_players[i] in player_options:
                        default_index_player = player_options.index(default_offense_players[i])
                    else:
                        default_index_player = 0  
                        st.warning(f"Default player '{default_offense_players[i]}' for Offense Player {i + 1} not found. Defaulting to the first available player.")
                    
                    player = st.selectbox(
                        f"Player {i + 1}",
                        options=player_options,
                        index=default_index_player,
                        key=f"offense_player_{i}",
                    )
                
                with col_position:
                    if default_offense_positions[i] in unique_offense_position:
                        default_index_position = unique_offense_position.index(default_offense_positions[i])
                    else:
                        default_index_position = 0  
                        st.warning(f"Default position '{default_offense_positions[i]}' for Offense Player {i + 1} not found. Defaulting to the first available position.")
                    
                    position = st.selectbox(
                        f"Position",
                        options=unique_offense_position,
                        index=default_index_position,
                        key=f"offense_position_{i}",
                    )
                
                with col_route:
                    if default_offense_routes[i] in unique_routes:
                        default_index_route = unique_routes.index(default_offense_routes[i])
                    else:
                        default_index_route = 0 
                        st.warning(f"Default route '{default_offense_routes[i]}' for Offense Player {i + 1} not found. Defaulting to the first available route.")
                    
                    route = st.selectbox(
                        f"Route",
                        options=unique_routes,
                        index=default_index_route,
                        key=f"offense_route_{i}",
                    )

                    
                offense_selected_players.append(player)
                offense_selected_positions.append(position)
                offense_selected_routes.append(route)
            
            error_placeholder_offense = st.empty()
            if len(set(offense_selected_players)) != 11:
                error_placeholder_offense.error("Please select exactly 11 players for the offense.")
            elif len(offense_selected_players) > 11:
                error_placeholder_offense.error("Too many players selected. Please select exactly 11 players for the offense.")
            elif len(offense_selected_players) != len(set(offense_selected_players)):
                error_placeholder_offense.error("Each offense player must be unique!")
            else:
                error_placeholder_offense.empty()

            offense_ratings = get_weighted_player_rating(self.player_df,offense_selected_players, quarter, down)
        
        return offense_play_type, offense_team_name, off_score, off_timeout_remaining, offense_wp, offense_formation, offense_selected_players, offense_selected_positions, offense_selected_routes, offense_ratings, offense_team_name
    

    def defense_details(self, defense_col, quarter, down):
        defense_selected_players = []
        defense_selected_positions = []
        defense_selected_assignments = []
        unique_defense_position = [
            "DT", "DE", "NT", "OLB", "MLB", "ILB", "LB", "FS", "SS", "CB", "DB"
        ]
        unique_coverage =[
            "Contain", "Deep Run Support", "Run Contain", 
            "Cutback Contain", "General Run Defense", "Force", 
            "Gap Control", "Gap Fill", "Inside Gap",
            "Zone Coverage", "Hook Zone", "Flat Zone Right", 
            "Flat Zone Left", "Deep Zone", "Deep Third Right", 
            "HCR", "HCL", "CFL", "MAN", "HOL", "FR", "3M", "3R", "3L", 
            "2R", "2L", "FL", "Pass Rush", "4IR", "4OL", "4IL", "4OR", 
            "DF", "PRE", "Unknown Assignment", "CFR", "Run Support", 
            "Deep Third Right", "Flat Zone Right", "Flat Zone Left"
        ] 
        unique_defense_formation = [
            '4-3 Defense', '4-2-5 Nickel', 'Interior DL Heavy', '5-2 Defense',
            'Heavy LB Defense', '3-4 Defense', 'Big Secondary and LB',
            'Heavy DE Front', 'Secondary Emphasis with DT', 'Other',
            '3-1-7 Quarter', 'Big Nickel', 'LB Heavy Defense', '4-1-6 Dime'
        ]

        with defense_col:
            defense_col_a, defense_col_b = st.columns(2)

            with defense_col_a:
                st.subheader(":blue[Defense]",divider="gray")

            with defense_col_b:
                defesne_play_option_map = {0: ":material/conditions: Man", 1: ":material/detection_and_zone: Zone"}
                defense_play_selection = st.pills("Coverage", options=defesne_play_option_map.keys(), format_func=lambda option: defesne_play_option_map[option], selection_mode="single", default=1, key="defense_play_selection")
                defense_play_type = defesne_play_option_map[defense_play_selection]
                defense_play_type = defense_play_type.split(":")[-1].strip()

            defense_col_c, defense_col_d, defense_col_e, defense_col_f = st.columns(4)

            with defense_col_c:
                defense_team_name = st.selectbox("Defense Team", options=self.team_names, index=1, key="defense_team_name")

            with defense_col_d:
                def_score = st.number_input("Score Points", key="defense_score", min_value=0, max_value=99, step=1, value=3, format="%d")

            with defense_col_e:
                def_timeout_remaining = st.selectbox("Timeout Left", options=[3,2,1],key="def_timeout")

            with defense_col_f:
                defense_wp = st.number_input("Win Probablity",key="defense_wp", value=0.09794 ,format="%.5f")


            
            selected_defense_logo_path = self.logo_folder / f"{defense_team_name}.png"
            if selected_defense_logo_path.exists():
                img_base64 = self.img_to_base64(str(selected_defense_logo_path))
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
                        <img src="data:image/png;base64,{img_base64}" alt="{defense_team_name}" 
                            style="width: 250px; height: 250px; object-fit: contain;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning(f"Logo not found for team: {defense_team_name}")

                    
            defense_formation = st.selectbox("Defense Formation", options = unique_defense_formation, index=11)
            
            default_players = [
                "Stephon Gilmore", "Rodney McLeod", "DeForest Buckner", "Yannick Ngakoue",
                "Ifeadi Odenigbo", "Kenny Moore", "Zaire Franklin", "Brandon Facyson",
                "Bobby Okereke", "Kwity Paye", "Rodney Thomas"
            ]

            default_positions = [
                "CB", "FS", "DT", "DE", "DE", "CB",
                "OLB", "CB", "MLB", "DE", "FS"
            ]

            default_cover_assignments = [
                "MAN", "HCR", "Pass Rush", "Pass Rush", "Pass Rush",
                "CFL", "HCL", "3L", "CFR", "Pass Rush", "3M"
            ]

            for i in range(11):
                def_col_player, def_col_position, col_coverage = st.columns(3)
                
                with def_col_player:
                    player_options = self.player_df['displayName'].tolist()
                    
                    if default_players[i] in player_options:
                        default_index_player = player_options.index(default_players[i])
                    else:
                        default_index_player = 0  
                    
                    def_player = st.selectbox(
                        f"Player {i + 1}",
                        options=player_options,
                        index=default_index_player,
                        key=f"defense_player_{i}",
                
                    )
                
                with def_col_position:
                    if default_positions[i] in unique_defense_position:
                        default_index_position = unique_defense_position.index(default_positions[i])
                    else:
                        default_index_position = 0  
                    
                    def_position = st.selectbox(
                        f"Position",
                        options=unique_defense_position,
                        index=default_index_position,
                        key=f"defense_position_{i}",
                   
                    )
                
                with col_coverage:
                    if default_cover_assignments[i] in unique_coverage:
                        default_index_coverage = unique_coverage.index(default_cover_assignments[i])
                    else:
                        default_index_coverage = 0 
                    
                    def_coverage = st.selectbox(
                        f"Coverage",
                        options=unique_coverage,
                        index=default_index_coverage,
                        key=f"defense_coverage_{i}",
                    )
                    
                defense_selected_players.append(def_player)
                defense_selected_positions.append(def_position)
                defense_selected_assignments.append(def_coverage)

            error_placeholder_defense = st.empty()
            if len(set(defense_selected_players)) != 11:
                error_placeholder_defense.error("Please select exactly 11 players for the defense.")
            elif len(defense_selected_players) > 11:
                error_placeholder_defense.error("Too many players selected. Please select exactly 11 players for the defense.")
            elif len(defense_selected_players) != len(set(defense_selected_players)):
                error_placeholder_defense.error("Each defense player must be unique!")
            else:
                error_placeholder_defense.empty()

            defense_ratings = get_weighted_player_rating(self.player_df, defense_selected_players, quarter, down)

        return defense_play_type, defense_team_name, def_score, def_timeout_remaining, defense_wp, defense_formation, defense_selected_players, defense_selected_positions, defense_selected_assignments, defense_ratings, defense_team_name
    

    def create_donut_plot(self, probabilities):
        sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
        top_items = list(sorted_probabilities.items())[:3] 
        other_items = list(sorted_probabilities.items())[3:]
        
        names = [f"{item[0]} yards ({float(item[1]) * 100:.1f}%)" for item in top_items] + [
            f"Other yards ({sum(prob for _, prob in other_items) * 100:.1f}%)"
        ]
        values = [item[1] for item in top_items] + [sum(prob for _, prob in other_items)]
        
        colors = sns.color_palette("twilight_shifted", len(values))

        fig, ax = plt.subplots(figsize=(5, 5))

        wedges, _ = ax.pie(
            values, 
            colors=colors, 
            wedgeprops=dict(width=0.5, edgecolor='black')
        )
        
        my_circle = plt.Circle((0, 0), 0.7, color='white',ec='black', lw=1)
        ax.add_artist(my_circle)
        
        ax.text(0, 0.05, 'Predicted', fontsize=14, color='black', ha='center') 
        ax.text(0, -0.2 , 'Yards', fontsize=14, color='black', ha='center') 
        
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="gray", lw=0.5, alpha=0.8)
        
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = "left" if x >= 0 else "right"

            ax.annotate(
                names[i], 
                xy=(x, y), 
                xytext=(1.2 * np.sign(x), 1.2 * y),
                horizontalalignment=horizontalalignment, 
                bbox=bbox_props, 
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle=f"angle,angleA=0,angleB={ang}",
                    color='gray',
                    lw=1
                ),
                fontsize=8
            )

        plt.tight_layout()
        st.pyplot(fig)

    
    def plot_bar_with_percentages(self, probabilities):
        df = pd.DataFrame(list(probabilities.items()), columns=['strategy', 'percent'])
        df = df.sort_values(by='percent', ascending=False).reset_index(drop=True)
        top_3 = df.head(3)

        other_strategy = df.tail(df.shape[0] - 3)
        other_strategy_sum = other_strategy['percent'].sum()
        other_strategy_df = pd.DataFrame({
            'strategy': ['Other Strategy'],
            'percent': [other_strategy_sum]
        })

        df_combined = pd.concat([top_3, other_strategy_df], ignore_index=True)
        df_combined['percentage'] = df_combined['percent'] * 100

        fig, ax = plt.subplots(figsize=(10, 6))
       
        sns.barplot(
            x="percent", 
            y="strategy", 
            data=df_combined, 
            estimator=sum, 
            errorbar=None, 
            hue="strategy",
            ax=ax,
            legend=False,
            palette="Set3",
            width=0.65,
            hatch='/'
        )

        for index, row in df_combined.iterrows():
            ax.text(
                row['percent'] + max(df_combined['percent']) * 0.05, 
                index, 
                f"{row['percentage']:.2f}%",  
                color='black',
                va='center', 
                fontsize=11,
                bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1')
            )

        ax.set_xticks([])  
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['left'].set_color('k')
        ax.set_xlim(0, max(df_combined['percent']) * 1.2)
        ax.tick_params(axis='both', which='major', labelsize=17, width=2.5, length=10)
        for bar in ax.patches:
            bar.set_edgecolor('black') 
            bar.set_linewidth(1.2)
        plt.tight_layout()
        st.pyplot(fig)

    def show_offense_player_summary(self, df_pass, df_route):
        summary = df_pass[["passer_player_name", "receiver_player_name", "total_passes", "C %", "I %", "C right %", "C left %", "C middle %"]].copy()
        summary = summary[summary['total_passes'] > 1]
        summary['max_pct'] = summary[['C right %', 'C left %', 'C middle %']].max(axis=1)
        summary['preferred_side'] = summary[['C right %', 'C left %', 'C middle %']].idxmax(axis=1).str.replace('C ', '').str.replace(' %', '').str.capitalize()
        summary['preferred_location'] = summary['max_pct'].astype(str) + ' % on ' + summary['preferred_side']
        summary = summary.sort_values('max_pct', ascending=False)
        summary = summary.drop_duplicates(subset=['passer_player_name', 'receiver_player_name'], keep='first')
        summary = summary.drop(['max_pct', 'preferred_side',"C right %", "C left %", "C middle %"], axis=1)
        summary = summary.sort_values(by='total_passes', ascending=False)
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
                if self.name_match(display_name, receiver):
                    matched_display_name = display_name
                    break

            if matched_display_name:
                receiver_routes = df_route[df_route['displayName'] == matched_display_name]
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

            final_rows.append({
                'Passer - Receiver': f"{passer} - {receiver}",
                'Total Passes': total_passes,
                'Complete Pass': f"{round(c_percent, 2)} %",
                'Incomplete Pass': f"{round(i_percent, 2)} %",
                'Preferred Location': preferred_location,
                'Receiver Routes Ran %': routes_str,
            })

        final_df = pd.DataFrame(final_rows)
        return final_df
    

    def get_label_rotation(self, angle, offset):
        rotation = np.rad2deg(angle + offset)
        if angle <= np.pi:
            alignment = "right"
            rotation += 180
        else:
            alignment = "left"
        return rotation, alignment

    def add_labels(self, angles, values, labels, offset, ax):
        padding = 10
        for angle, value, label in zip(angles, values, labels):
            rotation, alignment = self.get_label_rotation(angle, offset)
            ax.text(
                x=angle,
                y=value + padding,
                s=label,
                ha=alignment,
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=10,
    
            )

    def process_and_plot_curcularbarplot(self,df, group_cols, value_col, category_label,circularbar_cmap_name, inner_radius=100, ax=None):
        outcomes = df.groupby(group_cols, dropna=True)[value_col].value_counts().unstack().reset_index()
        value_types = list(df[value_col].unique())
        temp_df = (
            outcomes.groupby(group_cols, dropna=True)[value_types].sum().reset_index()
        )
        temp_df["row_sum"] = temp_df.loc[:, value_types].sum(axis=1)
        temp_df[value_types] = temp_df[value_types].apply(lambda x: x / temp_df["row_sum"])
        
        circular_df = temp_df[[category_label, "row_sum", "field_zone"]]
        circular_df = circular_df.sort_values(by=["field_zone", category_label])
        print(circular_df)
        
        VALUES = circular_df["row_sum"].values
        LABELS = circular_df[category_label].values
        GROUP = circular_df["field_zone"].values
        PAD = 3
        ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
        ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
        WIDTH = (2 * np.pi) / len(ANGLES)
        GROUPS_SIZE = [len(i[1]) for i in circular_df.groupby("field_zone")]
        OFFSET = np.pi / 2
        offset = 0
        IDXS = []
        for size in GROUPS_SIZE:
            IDXS += list(range(offset + PAD, offset + size + PAD))
            offset += size + PAD

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        
        ax.set_theta_offset(OFFSET)
        ax.set_ylim(-100, inner_radius)
        ax.set_frame_on(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        unique_groups = np.unique(GROUP)
        group_names = {
            'A': 'Midfield',
            'B': 'Opponent Red Zone',
            'C': 'Own Deep Zone',
            'D': 'Own Territory'
        }
        palette = sns.color_palette(circularbar_cmap_name, n_colors=len(unique_groups))
        group_colors = {group: palette[i] for i, group in enumerate(unique_groups)}
        COLORS = [group_colors[group] for group in GROUP]
        
        ax.bar(
            ANGLES[IDXS],
            VALUES,
            width=WIDTH,
            color=COLORS,
            edgecolor="black",
            linewidth=1
        )
        
        self.add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)
        
        offset = 0
        for group, size in zip(["","","",""], GROUPS_SIZE):
            x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
            ax.plot(x1, [-5] * 50, color="#333333")
            ax.text(
                np.mean(x1),
                -20,
                group,
                color="#333333",
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center"
            )
            x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
            ax.plot(x2, [20] * 50, color="#bebebe", lw=0.8)
            ax.plot(x2, [40] * 50, color="#bebebe", lw=0.8)
            ax.plot(x2, [60] * 50, color="#bebebe", lw=0.8)
            offset += size + PAD
        
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=group_colors[group], edgecolor='white', label=group_names.get(group, group)) for group in unique_groups]
        legend = ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.01), ncol=len(unique_groups))
        legend.set_frame_on(False)

    def process_and_plot_heatmap(self, fig_width, fig_height, df, group_cols, value_col, category_label, heatmap_cmap_name, ax=None):
        outcomes = df.groupby(group_cols, dropna=True)[value_col].value_counts().unstack().reset_index()
        value_types = list(df[value_col].unique())
        temp_df = (
            outcomes.groupby(group_cols, dropna=True)[value_types].sum().reset_index()
        )
        temp_df["row_sum"] = temp_df.loc[:, value_types].sum(axis=1)
        temp_df[value_types] = temp_df[value_types].apply(lambda x: x / temp_df["row_sum"])
        
        melted_df = temp_df.melt(
            id_vars=group_cols[0],
            value_vars=value_types,
            var_name="field_zone",
            value_name="proportion"
        )
        categories = list(melted_df[group_cols[0]].unique())
        sorted_categories = sorted(categories, key=str.lower)
        melted_df[group_cols[0]] = pd.Categorical(
            melted_df[group_cols[0]],
            categories=sorted_categories,
            ordered=True
        )
        melted_df = melted_df.sort_values(group_cols[0], ascending=False)
        
        y_spacing_factor = 0.5
        x_spacing_factor = 0.5
        y_mapping = {category: i * y_spacing_factor for i, category in enumerate(sorted_categories)}
        melted_df['y_pos'] = melted_df[group_cols[0]].map(y_mapping)
        sorted_x_categories = sorted(melted_df["field_zone"].unique(), key=str.lower)
        x_mapping = {category: i * x_spacing_factor for i, category in enumerate(sorted_x_categories)}
        melted_df['x_pos'] = melted_df["field_zone"].map(x_mapping)
        
        cmap = plt.get_cmap(heatmap_cmap_name)
        if ax is None:
            _, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        for value_type in value_types:
            d = melted_df[melted_df["field_zone"] == value_type]
            y = d['y_pos']
            x = d['x_pos']
            color = cmap(d["proportion"])
            ax.scatter(x, y, color=color, s=250, edgecolor="k")
        
        ax.set_frame_on(False)
        ax.grid(which='major', color='#CCCCCC', linestyle='--',alpha=0.8)
        ax.grid(which='minor', color='#CCCCCC', linestyle=':',alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks([x_mapping[cat] for cat in sorted_x_categories])
        ax.set_xticklabels(sorted_x_categories, ha='right')
        ax.set_yticks([y_mapping[cat] for cat in sorted_categories])
        ax.set_yticklabels(sorted_categories)
        ax.set_ylim(-y_spacing_factor, (len(sorted_categories) - 1) * y_spacing_factor + y_spacing_factor / 2)
        ax.set_xlim(-x_spacing_factor, (len(sorted_x_categories) - 1) * x_spacing_factor + x_spacing_factor / 2)
        ax.tick_params(axis='both', which='both', length=0, colors="0.3")
        ax.set_xlabel(value_col, loc="right")

    def plot_combined(self,fig_width, fig_height, df,group_cols,value_col,category_label,heatmap_cmap_name="BuGn",circularbar_cmap_name="Pastel2",inner_radius=100, main_width=20, main_height=8 ):
        fig = plt.figure(figsize=(main_width, main_height))
        gs = gridspec.GridSpec(1, 3, width_ratios=[.1, 0.5, 0.1]) 
        ax_heatmap = fig.add_subplot(gs[0])
        self.process_and_plot_heatmap(
            fig_width=fig_width,
            fig_height=fig_height,
            df=df,
            group_cols=group_cols,
            value_col=value_col,
            category_label=category_label,
            heatmap_cmap_name=heatmap_cmap_name,
            ax=ax_heatmap
        )

        ax_circular = fig.add_subplot(gs[1], projection='polar')
        self.process_and_plot_curcularbarplot(
            df=df,
            group_cols=group_cols,
            value_col=value_col,
            category_label=category_label,
            ax=ax_circular,
            circularbar_cmap_name=circularbar_cmap_name,
            inner_radius=inner_radius
        )

        plt.tight_layout()
        st.pyplot(fig)


    def run_streamlit_app(self):
        st.title("Welcome to :orange[Playground Simulator]")
        st.subheader(":violet[Game Situation]",divider="gray")    
        quarter, down, yards_to_go, yards_to_endzone, game_half, is_second_half, quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining = self.game_situation_components()
        offense_col, defense_col = st.columns(2)
        offense_play_type, offense_teaam_name, off_score, off_timeout_remaining, offense_wp, offense_formation, offense_selected_players, offense_selected_positions, offense_selected_routes, offense_ratings, offense_team_name = self.offense_details(offense_col, quarter, down)
        defense_play_type, defense_team_name, def_score, def_timeout_remaining, defense_wp, defense_formation, defense_selected_players, defense_selected_positions, defense_selected_assignments, defense_ratings, defense_team_name = self.defense_details(defense_col, quarter, down)

        if 'submit_button' not in st.session_state:
            st.session_state.submit_button = False

        st.markdown(
            """
            <style>
            .stButton > button { color: white; background-color: #76528BFF; padding: 10px; font-size: 16px; text-align: center; border: none; border-radius: 8px; }
            .stButton > button:hover {background-color: #603F83FF;}
            </style>
            """,
            unsafe_allow_html=True
        )

        input_valid = True
        st.session_state.submit_button = False

        if len(set(offense_selected_players)) != 11:
            st.error("You must select exactly 11 unique offense players.")
            input_valid = False

        if len(set(defense_selected_players)) != 11:
            st.error("You must select exactly 11 unique defense players.")
            input_valid = False

        if set(offense_selected_players) & set(defense_selected_players):
            st.error("Players cannot overlap between offense and defense.")
            input_valid = False

        if set(offense_teaam_name) == set(defense_team_name):
            st.error("Teams cannot overlap between offense and defense.")
            input_valid = False


        clicked = st.button("Predict Game Play Scenario", key="predict_button", use_container_width=True, type="primary")

        if clicked and input_valid:
            st.session_state.submit_button = True

        whole_offense_ratings = round(np.mean(offense_ratings), 6)
        whole_defense_ratings = round(np.mean(defense_ratings), 6)

        if st.session_state.submit_button and input_valid:
            pay_load = self.genrate_payload(
                quarter, down, yards_to_go, yards_to_endzone, game_half, is_second_half, 
                quarter_seconds_remaining, half_seconds_remaining, game_seconds_remaining,
                offense_play_type, off_score, off_timeout_remaining, 
                offense_wp, offense_formation, offense_selected_positions, 
                offense_selected_routes, whole_offense_ratings,
                defense_play_type, def_score, def_timeout_remaining, defense_wp, 
                defense_formation, defense_selected_positions, 
                defense_selected_assignments, whole_defense_ratings
            )
            categorical_features = [
                'quarter', 'down', 'offenseFormation', 'play_type', 
                'defenseFormation', 'pff_manZone', 'game_half', 'is_second_half',
                'offense_defense_interaction'
            ]
            numerical_features = [
                'yardsToGo', 'absoluteYardlineNumber', 'average_offense_rating', 
                'average_defense_rating', 'quarter_seconds_remaining', 
                'half_seconds_remaining', 'game_seconds_remaining', 
                'score_differential', 'wp', 'def_wp', 
                'down_yardsToGo', 'quarter_time_ratio', 'half_time_ratio', 
                'posteam_timeout_usage', 'defteam_timeout_usage', 'normalized_down', 
                'normalized_quarter', 'normalized_time_remaining', 'situation_pressure', 
                'goal_to_go', 'blocking_ratio', 'routes_ratio', 'special_actions_ratio', 
                'no_assignment_ratio', 'run_defense_ratio', 'zone_defense_ratio', 
                'man_defense_ratio', 'pass_rush_ratio', 'miscellaneous_ratio', 
                'special_defense_ratio', 'skill_players_count', 'skill_players_ratio',
                'offensive_line_count', 'offensive_line_ratio', 'uncommon_positions_count', 
                'uncommon_positions_ratio', 'defensive_line_count', 'defensive_line_ratio',
                'linebackers_count', 'linebackers_ratio', 'defensive_backs_count', 
                'defensive_backs_ratio', 'non_traditional_count', 'non_traditional_ratio'
            ]
            
            probabilities = predict_yard_bins_sorted(
                game_scenario=pay_load,
                model=model, 
                cat_features=categorical_features, 
                cat_mapping=cat_mapping, 
                numerical_features=numerical_features, 
                scaler=scaler, 
                y_labels=y_labels
            )

            top_items = list(probabilities.items())[:3] 
            other_items = list(probabilities.items())[3:] 
            top_bin_1, top_prob_1 = top_items[0][0], top_items[0][1] if len(top_items) > 0 else ("N/A", 0)
            top_bin_2, top_prob_2 = top_items[1][0], top_items[1][1] if len(top_items) > 1 else ("N/A", 0)
            top_bin_3, top_prob_3 = top_items[2][0], top_items[2][1] if len(top_items) > 2 else ("N/A", 0)
            other_prob = sum(prob for _, prob in other_items) if len(other_items) > 0 else 0

            donut_prediction_text = f"""
                1. **{top_bin_1} yards**: This is the most probable outcome, with a likelihood of **{top_prob_1:.2%}**.
                2. **{top_bin_2} yards**: This outcome follows with a probability of **{top_prob_2:.2%}**.
                3. **{top_bin_3} yards**: A less likely outcome, predicted with a probability of **{top_prob_3:.2%}**.
                4. **Other yards**: Remaining outcomes have a combined probability of **{other_prob:.2%}**.

                    {f"- **High Confidence Play**: The model strongly favors a gain of **{top_bin_1} yards**, which aligns with offensive strengths." if top_prob_1 > 0.5 else ""}
                    {f"- **Probable Outcomes**: Based on testing to unseen data, (**{top_bin_1} yards** and **{top_bin_2} yards**) are the most likely to occur. While outcomes involving **{top_bin_3} yards** are possible, but they occur rarely." if top_prob_1 > top_prob_2 * 0.8 else ""}

            """
            
            first_key = list(probabilities.keys())[0]  
            pay_load['yardsGained'] = first_key
            categorical_features.append('yardsGained')

            startegy = predict_strategy_sorted(
                game_scenario=pay_load,
                model=offense_strategy_model, 
                cat_features=categorical_features, 
                cat_mapping=offense_strategy_cat_mapping, 
                numerical_features=numerical_features, 
                scaler=offense_strategy_scaler, 
                y_labels=offense_strategy_y_labels
            )
            startegy_top_items = list(startegy.items())[:3] 
            startegy_other_items = list(startegy.items())[3:] 
            startegy_top_bin_1, startegy_top_prob_1 = startegy_top_items[0][0], startegy_top_items[0][1] if len(startegy_top_items) > 0 else ("N/A", 0)
            startegy_top_bin_2, startegy_top_prob_2 = startegy_top_items[1][0], startegy_top_items[1][1] if len(startegy_top_items) > 1 else ("N/A", 0)
            startegy_top_bin_3, startegy_top_prob_3 = startegy_top_items[2][0], startegy_top_items[2][1] if len(startegy_top_items) > 2 else ("N/A", 0)
            startegy_other_prob = sum(startegy_prob for _, startegy_prob in startegy_other_items) if len(startegy_other_items) > 0 else 0


            strategy_prediction_text = f"""
                1. **{startegy_top_bin_1}**: This is the most probable strategy, with a likelihood of **{startegy_top_prob_1:.2%}**.
                2. **{startegy_top_bin_2}**: This strategy follows with a probability of **{startegy_top_prob_2:.2%}**.
                3. **{startegy_top_bin_3}**: A less likely strategy, predicted with a probability of **{startegy_top_prob_3:.2%}**.
                4. **Other Strategies**: Remaining strategies have a combined probability of **{startegy_other_prob:.2%}**.

                    {f"- **High Confidence Strategy**: Given the current situation, employing **{startegy_top_bin_1}** is advantageous as it leverages the team's offensive strengths." if startegy_top_prob_1 > 0.5 else ""}
                    {f"- **Probable Strategies**: Based on testing on unseen data, **{startegy_top_bin_1}** and **{startegy_top_bin_2}** are the most likely to be executed. While **{startegy_top_bin_3}** is possible, it occurs rarely." if startegy_top_prob_1 > startegy_top_prob_2 * 0.8 else ""}

            """
  
            summary_intro_text = f"""   
                ### **Perfectly Imperfect: Game Analysis, Predictive Summaries, Tendencies and Player Performance Insights**

                This analysis provides a comprehensive overview of the current game situation, player performance, predicted yard gain outcomes, predicted offensive strategies & offering insights into both offensive and defensive tendencies. Every insight is tailored to empower coaches with data-driven decision:
            """

            st.markdown(summary_intro_text)
            st.markdown("<h3 style='text-align: center; color: grey;'>Tensorflow Predictive Insights: Yard Gains and Offensive Strategies</h3>", unsafe_allow_html=True)

            if offense_play_type == "run":
                self.create_donut_plot(probabilities)
                st.subheader("Run Play Type Report is in Progress & will be continued in future......")

            if offense_play_type == "pass":
                yards_prob, strategy_prob = st.columns(2)
                with yards_prob:
                    self.create_donut_plot(probabilities)

                with strategy_prob:
                    self.plot_bar_with_percentages(startegy)
                    
                prediction_text = f"""
                ---

                Based on the current game scenario and player configurations, the predictive model has evaluated the likelihood of yard gains and an Offensive Strategy based on the current game scenario *(Down: **{down}**, Distance: **{yards_to_go}** yards, Quarter: **{quarter}**, Time Remaining: **{quarter_seconds_remaining}** seconds, Score Difference: **{pay_load.get("score_differential")}** points)* and player configurations. The model suggests:
                """
                st.markdown(prediction_text)

                donut_pred_text, startegy_pred_text = st.columns(2)
                with donut_pred_text:
                    st.markdown("<h5 style='text-align: center; color: grey;'>Yard Gains Insights</h5>", unsafe_allow_html=True)
                    st.markdown(donut_prediction_text)

                with startegy_pred_text:
                    st.markdown("<h5 style='text-align: center; color: grey;'>Offensive Strategies Insights</h5>", unsafe_allow_html=True)
                    st.markdown(strategy_prediction_text)

                st.divider()
                offense, defense = st.columns(2)
                with offense:
                    st.header(":green[Offense]")
                    selected_team_logo_path = self.logo_folder / f"{offense_team_name}.png"
                    if selected_team_logo_path.exists():
                        img_base64 = self.img_to_base64(str(selected_team_logo_path))
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
                                <img src="data:image/png;base64,{img_base64}" alt="{offense_team_name}" 
                                    style="width: 250px; height: 250px; object-fit: contain;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        offense_team_full_name = self.team_full_names[offense_team_name]
                        st.markdown(f"<h1 style='text-align: center; color: gray;'>{offense_team_full_name}</h1>", unsafe_allow_html=True)
                    else:
                        st.warning(f"Logo not found for team: {offense_team_name}")
            

                with defense:
                    st.header(":blue[Defense]")
                    selected_defense_logo_path = self.logo_folder / f"{defense_team_name}.png"
                    if selected_defense_logo_path.exists():
                        img_base64 = self.img_to_base64(str(selected_defense_logo_path))
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: center; align-items: center; margin: 20px;">
                                <img src="data:image/png;base64,{img_base64}" alt="{defense_team_name}" 
                                    style="width: 250px; height: 250px; object-fit: contain;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        defense_team_full_name = self.team_full_names[defense_team_name]
                        st.markdown(f"<h1 style='text-align: center; color: gray;'>{defense_team_full_name}</h1>", unsafe_allow_html=True)
                    else:
                        st.warning(f"Logo not found for team: {defense_team_name}")

                st.divider()

                game_situation_text_1, game_situation_text_2 = st.columns(2)
                with game_situation_text_1:
                    game_situation_text = f"""
        
                    ### **Game :orange[Situational] Context**
                    - **Quarter**: The game is in the `{quarter}` quarter, with the `{down}` down being played.
                    - **Yards to Go**: The offense needs `{yards_to_go}` yards to secure a first down.
                    - **Field Position**: The ball is positioned `{yards_to_endzone}` yards from the endzone.
                    - **Game Clock**: There are `{quarter_seconds_remaining}` seconds left in this quarter, `{half_seconds_remaining}` seconds in the half, and `{game_seconds_remaining}` seconds remaining in the game overall.
                    - **Game Segment**: The play is taking place in `{game_half.split(":")[-1]}`.
                        {f"- **Critical Situation**: With less than 2 minutes remaining in the game, every play becomes vital for both teams." if game_seconds_remaining <= 120 else ""}
                        {f"- **Red Zone Alert**: The ball is within the 20-yard line, increasing scoring opportunities." if yards_to_endzone <= 20 else ""}
                        {f"- **Third Down Pressure**: A critical third down is in play, and the offense must execute effectively to avoid a punt or turnover." if down == 3 else ""}
                    """
                    st.markdown(game_situation_text)

                with game_situation_text_2:
                    summary_data = {
                            "Aspect": [
                                "Formation",
                                "Play Type",
                                "Average Player Ratings",
                                "Timeouts Remaining",
                                "Key Situations",
                                "Win Probability (WP)",
                                "Key Strength"
                            ],
                            "Offense": [
                                offense_formation,
                                offense_play_type, 
                                f"{round(np.mean(offense_ratings), 2)}",
                                f"{off_timeout_remaining}",
                                f"Critical Red Zone Opportunity" if yards_to_endzone <= 20 else "Midfield Setup" if 20 < yards_to_endzone <= 50 else "Deep Field Challenge",
                                f"{offense_wp:.2%}",
                                "Skill-heavy explosive potential" if pay_load.get('skill_players_ratio', 0) > 0.6 else "Balanced play flexibility" if 0.4 < pay_load.get('skill_players_ratio', 0) <= 0.6 else "Run-oriented focus"
                            ],
                            "Defense": [
                                defense_formation,
                                defense_play_type,
                                f"{round(np.mean(defense_ratings), 2)}",
                                f"{def_timeout_remaining}",
                                f"Defend Red Zone" if yards_to_endzone <= 20 else "Prevent Big Gains" if 20 < yards_to_endzone <= 50 else "Control Field Position",
                                f"{defense_wp:.2%}",
                                "Aggressive pass rush" if pay_load.get("pass_rush_ratio", 0) > 0.4 else "Zone containment focus" if pay_load.get("zone_defense_ratio", 0) > 0.5 else "Balanced coverage"
                            ],
                            
                    }

                    # Convert to DataFrame
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)

                offense_player_data, defense_player_data = st.columns(2)

                with offense_player_data:
                    st.subheader(":red[Offense Players]",divider="gray")
                    offense_data = {
                        "Name": offense_selected_players,
                        'Position': offense_selected_positions,
                        'Route': offense_selected_routes,
                        'Rating': offense_ratings
                    }
                    offense_data_df = pd.DataFrame(offense_data)
                    st.dataframe(offense_data_df,height=423, use_container_width=True, hide_index=True)
                    
                    offense_player_selection_text_continue = f"""
                        {f"- **High Leverage Situation**: With only {yards_to_go} yards to go, the offense has a strong chance to convert and sustain the drive." if yards_to_go <= 5 else ""}
                        {f"- **Deep Field Challenge**: Positioned {yards_to_endzone} yards from the endzone, the offense must strategize for significant yardage." if yards_to_endzone >= 50 else ""}
                        - **Skill Players**: {pay_load.get('skill_players_ratio', 0):.2%} of the players are in skill positions (e.g., QB, WR, TE).
                        - **Offensive Line**: {pay_load.get('offensive_line_ratio', 0):.2%} of the players are linemen.
                        - **Uncommon Positions**: {pay_load.get('uncommon_positions_ratio', 0):.2%} of the players are in less common positions.
                        {f"- **Skill-Heavy Formation**: An abundance of skill players indicates a focus on explosive plays." if pay_load.get('skill_players_ratio', 0) > 0.6 else ""}
                        - **Blocking Routes**: {pay_load.get('blocking_ratio', 0):.2%} of the players are involved in blocking assignments.
                        - **Pass Routes**: {pay_load.get('routes_ratio', 0):.2%} of the players are focused on route-running.
                        - **Special Actions**: {pay_load.get('special_actions_ratio', 0):.2%} of the players are executing special actions (e.g., Kneels, Spikes).
                        - **Unassigned Routes**: {pay_load.get('no_assignment_ratio', 0):.2%} of the players have no specific assignments.
                        {f"- **Pass-Heavy Formation**: A high proportion of players are running pass routes, emphasizing aerial attacks." if pay_load.get('routes_ratio', 0) > 0.5 else ""}
                        {f"- **Run-Heavy Formation**: Blocking assignments dominate, indicating a ground-oriented strategy." if pay_load.get('blocking_ratio', 0) > 0.5 else ""}
                        {f"- **Balanced Approach**: The offense appears to have a balanced mix of run and pass assignments." if 0.4 < pay_load.get('blocking_ratio', 0) < 0.6 else ""}
                    """
                    st.markdown(offense_player_selection_text_continue)

                with defense_player_data:
                    st.subheader(":blue[Defense Players]",divider="gray")
                    defense_data = {
                        "Name": defense_selected_players,
                        'Position': defense_selected_positions,
                        'Coverage': defense_selected_assignments,
                        'Rating': defense_ratings
                    }
                    defense_data_df = pd.DataFrame(defense_data)
                
                    st.dataframe(defense_data_df, height=423,use_container_width=True, hide_index=True)
                    defense_player_selection_text_continue = f"""
                        {f"- **Prevent Defense**: The defense is likely focused on stopping deep plays as the offense nears the endzone." if yards_to_endzone <= 20 else ""}
                        {f"- **Blitz Opportunity**: On third down, the defense might consider an aggressive blitz to force a turnover or punt." if down == 3 else ""}
                        {f"- **Clock Management**: With limited time remaining, the defense needs to balance aggression with preventing big plays." if game_seconds_remaining <= 120 else ""}
                        - **Run Defense**: {pay_load.get('run_defense_ratio', 0):.2%} of the players are assigned to stop the run.
                        - **Zone Coverage**: {pay_load.get('zone_defense_ratio', 0):.2%} of the players are covering zones.
                        - **Man Coverage**: {pay_load.get('man_defense_ratio', 0):.2%} of the players are in man-to-man coverage.
                        - **Pass Rush**: {pay_load.get('pass_rush_ratio', 0):.2%} of the players are focused on rushing the passer.
                        - **Miscellaneous Assignments**: {pay_load.get('miscellaneous_ratio', 0):.2%} are handling unique roles.
                        {f"- **Aggressive Coverage**: Pass rush strategies dominate, potentially forcing quick decisions from the QB." if pay_load.get('pass_rush_ratio', 0) > 0.4 else ""}
                        {f"- **Zone Dominance**: The defense is relying heavily on zone coverage, prioritizing area-based containment." if pay_load.get('zone_defense_ratio', 0) > 0.5 else ""}
                        {f"- **Man Coverage Strength**: A significant proportion of players are in man-to-man assignments, emphasizing tight coverage." if pay_load.get('man_defense_ratio', 0) > 0.4 else ""}
                        - **Defensive Line**: {pay_load.get('defensive_line_ratio', 0):.2%} of the players are in the defensive front.
                        - **Linebackers**: {pay_load.get('linebackers_ratio', 0):.2%} of the players are in the second level.
                        - **Defensive Backs**: {pay_load.get('defensive_backs_ratio', 0):.2%} of the players are in the secondary.
                        - **Non-Traditional Assignments**: {pay_load.get('non_traditional_ratio', 0):.2%} of the players have unconventional assignments.
                        {f"- **Pressure Formation**: A high ratio of defensive linemen suggests an emphasis on pressuring the QB." if pay_load.get('defensive_line_ratio', 0) > 0.4 else ""}
                        {f"- **Coverage Formation**: The secondary is heavily populated, prioritizing coverage over pressure." if pay_load.get('defensive_backs_ratio', 0) > 0.5 else ""}
                    """
                    st.markdown(defense_player_selection_text_continue)

                st.divider()
                st.subheader("Offense :violet[Pass-Receiver] Tendencies")
                st.markdown("""
                    This data provides insights into the connection between quarterbacks and their receivers, highlighting total passes, completion and incompletion rates, preferred field locations (e.g., **Left, Right, Middle**) 
                    for successful passes, and the receiver's route-running tendencies (e.g., **Go, Corner, Hitch**). It helps coaches understand where the passer-receiver duo excels, which areas or routes are most effective, 
                    and where improvements can be made to optimize offensive play design.
                    """
                )
                my_df = self.show_offense_player_summary(self.pass_receiver_df,self.routes_df)
                st.dataframe(my_df, hide_index=True, use_container_width=True,height=385)
                st.markdown(
                    """
                    #### Takeaway
                    This data provides actionable insights into how quarterbacks and receivers interact on the field. It helps answer:
                    - Which passer-receiver pairings are most reliable?
                    - Where on the field are they most effective?
                    - What routes should be emphasized to align with receiver tendencies?

                    By using this information, we can refine offensive strategies, design more effective plays, and strengthen their team’s passing game.
                    """
                )
                st.divider()
                st.markdown("<h3 style='text-align: center; color: gray;'>Defensive Coverage Schemes and Field Zones</h3>", unsafe_allow_html=True)
                st.markdown(
                    """
                    This visualization combines two types of charts—a **Heatmap** and a **Circular Bar Plot**—to analyze offensive plays and how all 31 NFL teams employed their defensive 
                    pass coverage schemes (`pff_passCoverage`) against them. The goal is to understand how different defensive strategies impacted pass outcomes (`passResult`) such as 
                    completions, incompletions, sacks, interceptions, or scrambles, across four key field zones (Opponent Red Zone, Midfield, Own Territory, Own Deep Zone)
                    """

                )
                pff_passCoverage_heatmap_text , pff_passCoverage_plot_text = st.columns(2)
                with pff_passCoverage_heatmap_text:
                    st.markdown(
                        """
                        #### Heatmap
                        **What It Shows:**
                        - The heatmap shows the proportion of pass outcomes (`passResult`) for different combinations of field zones and defensive coverage schemes.
                            - **X-axis:** Pass outcomes (`passResult`) such as Complete, Incomplete, Sack, Intercepted.
                            - **Y-axis:** Defensive pass coverage schemes.
                        - **Darker colors:** Indicate where certain outcomes are more frequent:
                            - Darker near **Complete:** The offense is effective against this coverage in the corresponding field zone & vice versa for Lighter or transparent shade.
                 
                        **Why It Matters:**
                        - Highlights where offenses excel or falter against specific defensive coverages.
                        - Identifies the effectiveness of defensive schemes in disrupting offensive plays in different field zones.

                        **Key Insight:**
                        - This heatmap helps pinpoint which defensive strategies were most effective in disrupting offensive plays across different field zones.
                        - A darker color near "**Complete**" suggests the offense thrives against that coverage in certain zones. A darker color near "**Incomplete**" or "**Intercepted**" points to defensive schemes that are highly effective in shutting down the offense.


                        """
                    )
                with pff_passCoverage_plot_text:
                    st.markdown(
                        """
                        #### Circular Bar Plot (Radial Bar Chart)
                        **What It Shows:**
                        - The circular chart is divided into four sections, each representing a field zone:
                            - **Opponent Red Zone** (0-25 yards)
                            - **Midfield** (26-50 yards)
                            - **Own Territory** (51-75 yards)
                            - **Own Deep Zone** (76-100 yards).
                        - Each bar in the chart represents a defensive pass coverage scheme (e.g., Cover - 0, Cover - 1, etc ) used in that field zone.
                        - The length of the bar shows how often that coverage was employed.

                        **Why It Matters:**
                        - **Peak of the Bar:** Longer bars highlight defensive schemes that were frequently used and potentially more effective against offensive plays in that zone.
                        - **Color:** Colors differentiate field zones, making it easier to compare defensive strategies across areas of the field.

                        **Key Insight:**
                        - Use this chart to identify field zones where the offense struggled most against certain defensive coverages. For example, the chart might reveal that offenses completed fewer passes in the **Opponent Red Zone** when faced with Zone coverage.

                        """
                    )
                
                self.plot_combined(
                    fig_width=3,
                    fig_height=8,
                    df=self.combo_df,
                    group_cols=["pff_passCoverage", "field_zone"],
                    value_col="passResult",
                    category_label="pff_passCoverage"
                )
                st.markdown(
                    """
                    #### Visualization Summary
                    - The **Heatmap** provides a big-picture view of how defensive coverage schemes influenced pass outcomes across the field.
                    - The **Circular Bar Plot** adds context by showing where defensive schemes are most prevalent and their general impact.

                    **How to Use It:**
                    - Start with the heatmap to identify offensive performance trends (e.g., dark colors for "Complete" or "Incomplete").
                    - Use the circular bar plot to dig deeper into the details, such as which defensive schemes were most frequently used and where offenses struggled the most.

                    **Takeaway:**
                    This visualization provides a clear breakdown of offensive performance against defensive strategies across key field zones. It helps answer:
                    - Where does your offense excel or struggle against specific coverage types?
                    - Which field zones present the biggest challenges for your plays?
                    - How can you adapt your offense to exploit weaknesses in defensive schemes?

                    By combining the heatmap with the circular bar plot, we can develop strategies to counter defensive coverage and optimize offensive performance.

                    """
                )
                st.divider()
                st.markdown("<h3 style='text-align: center; color: gray;'>Offensive Strategy and Field Zones</h3>", unsafe_allow_html=True)
                st.markdown(
                    """
                    This visualization combines two types of charts—a **Heatmap** and a **Circular Bar Plot**—to analyze offensive plays and the strategies (`strategy`) 
                    used by the offense in various field zones. The goal is to understand how different offensive strategies impacted pass outcomes (`passResult`), 
                    such as completions, incompletions, sacks, interceptions, or scrambles, across four key field zones (Opponent Red Zone, Midfield, Own Territory, Own Deep Zone).
                    """
                )
                startegy_heatmap_text , startegy_plot_text = st.columns(2)

                with startegy_heatmap_text:
                    st.markdown(
                        """
                        #### Heatmap
                        **What It Shows:**
                        - The heatmap shows the proportion of pass outcomes (`passResult`) for different combinations of field zones and offensive strategies.
                            - **X-axis:** Pass outcomes (`passResult`) such as Complete, Incomplete, Sack, Intercepted.
                            - **Y-axis:** Offensive strategies (`strategy`), such as Short Left Pass, Short Right Pass, Shotgun Isolation, or Deep Attack.
                        - **Darker colors:** Indicate where certain outcomes are more frequent:
                            - Darker near **Complete:** The offense was highly effective using this strategy in the corresponding field zone. Lighter or transparent shades indicate less frequent or less effective outcomes.

                        **Why It Matters:**
                        - Highlights where offenses excel or struggle with specific strategies in different field zones.
                        - Identifies which strategies lead to successful or unsuccessful outcomes.

                        **Key Insight:**
                        - This heatmap helps pinpoint offensive strategies that worked best in specific field zones.
                        - For example, a darker shade near "**Complete**" suggests that a strategy like Short Right Pass was highly effective in a specific zone.
                        - Conversely, darker shades near "**Incomplete**" or "**Intercepted**" indicate where the strategy failed or the defense disrupted the play.

                        """
                    )

                with startegy_plot_text:
                    st.markdown(
                        """ 
                        #### Circular Bar Plot (Radial Bar Chart)
                        **What It Shows:**
                        - The circular chart is divided into four sections, each representing a field zone:
                        - **Opponent Red Zone** (0-25 yards)
                        - **Midfield** (26-50 yards)
                        - **Own Territory** (51-75 yards)
                        - **Own Deep Zone** (76-100 yards).
                        - Each bar in the chart represents an offensive strategy (`strategy`) used in that field zone (e.g., Short Left Pass, Shotgun Isolation).
                        - The length of the bar shows how frequently that strategy was employed in that field zone.

                        **Why It Matters:**
                        - **Peak of the Bar:** Longer bars highlight strategies that were frequently used in specific field zones.
                        - **Color:** Colors differentiate field zones, making it easier to compare offensive strategy usage across the field.

                        **Key Insight:**
                        - Use this chart to identify which strategies are most commonly used in different field zones.
                        - For example, the chart might reveal that Short Left Pass is frequently used in the **Midfield**, while Short Middle Pass is less common in the **Opponent Red Zone**.

                        """
                    )

                self.plot_combined(
                    fig_width=4,
                    fig_height=10,
                    df=self.combo_df,
                    group_cols=["strategy", "field_zone"],
                    value_col="passResult",
                    category_label="strategy",
                    inner_radius=180,
                    main_width=20,
                    main_height=10
                )
                st.markdown(
                    """
                    #### Visualization Summary
                    - The **Heatmap** provides a big-picture view of how offensive strategies influenced pass outcomes across the field.
                    - The **Circular Bar Plot** adds context by showing which offensive strategies were most frequently used in different field zones.

                    **How to Use It:**
                    - Start with the heatmap to identify performance trends for specific strategies (e.g., dark colors for "Complete" or "Incomplete").
                    - Use the circular bar plot to explore the frequency of each strategy in various field zones and correlate it with the heatmap trends.

                    **Takeaway:**
                    This visualization provides a clear breakdown of offensive performance and strategy usage across key field zones. It helps answer:
                    - Which offensive strategies are most effective in specific field zones?
                    - Where do offensive strategies falter or succeed based on pass outcomes?
                    - How can you adjust your play-calling to maximize success in each field zone?

                    By combining the heatmap with the circular bar plot, we can develop strategies to optimize offensive plays and counter defensive responses effectively.

                    """
                )
                st.divider()
                st.markdown("<h3 style='text-align: center; color: gray;'>Defensive Formation and Field Zones</h3>", unsafe_allow_html=True)
                st.markdown(
                    """
                    This visualization combines two types of charts—a **Heatmap** and a **Circular Bar Plot**—to analyze how different defensive formations (`defenseFormation`) 
                    were used by all 31 NFL teams against offensive plays. The goal is to understand how these formations impacted pass outcomes (`passResult`), 
                    such as completions, incompletions, sacks, interceptions, or scrambles, across four key field zones (Opponent Red Zone, Midfield, Own Territory, Own Deep Zone).
                    """
                )
                defense_formation_heatmap_text , defense_formation_plot_text = st.columns(2)
                with defense_formation_heatmap_text:
                    st.markdown(
                        """
                        #### Heatmap
                        **What It Shows:**
                        - The heatmap displays the proportion of pass outcomes (`passResult`) for different combinations of field zones and defensive formations.
                        - **X-axis:** Pass outcomes (`passResult`) such as Complete, Incomplete, Sack, Intercepted.
                        - **Y-axis:** Defensive formations (`defenseFormation`), such as 3-4 Defense, Big Nickel, or 5-2 Defense.
                        - **Darker colors:** Indicate where certain outcomes are more frequent:
                        - Darker near **Incomplete** or **Intercepted:** Suggests the defensive formation was more effective in disrupting the offense in the corresponding field zone.
                        - Lighter or transparent colors indicate less frequent or less impactful combinations of outcomes and formations.

                        **Why It Matters:**
                        - Highlights where defensive formations were most effective at stopping or disrupting offensive plays.
                        - Identifies formations and zones where offenses managed to succeed or where defenses dominated.

                        **Key Insight:**
                        - This heatmap helps identify defensive formations that worked best in specific field zones.
                        - For example, a darker shade near "**Incomplete**" or "**Intercepted**" suggests the formation was effective in that zone.
                        - A darker shade near "**Complete**" indicates where the defense struggled against the offense.

                        """
                    )

                with defense_formation_plot_text:
                    st.markdown(
                        """
                        #### Circular Bar Plot (Radial Bar Chart)
                        **What It Shows:**
                        - The circular chart is divided into four sections, each representing a field zone:
                        - **Opponent Red Zone** (0-25 yards)
                        - **Midfield** (26-50 yards)
                        - **Own Territory** (51-75 yards)
                        - **Own Deep Zone** (76-100 yards).
                        - Each bar represents a defensive formation (`defenseFormation`) used in that field zone (e.g., 3-4 Defense, Big Nickel, or 5-2 Defense).
                        - The length of the bar shows how frequently that formation was used by the 31 teams in that field zone.

                        **Why It Matters:**
                        - **Peak of the Bar:** Longer bars highlight formations that were frequently employed in specific zones.
                        - **Color:** Colors differentiate field zones, making it easier to compare defensive formation usage across areas of the field.

                        **Key Insight:**
                        - Use this chart to identify field zones where certain defensive formations were heavily used.
                        - For example, the chart might show that 3-4 Defense was commonly used in the **Midfield**, while Big Nickel appeared more often in the **Opponent Red Zone**.

                        """
                    )

                self.plot_combined(
                    fig_width=3,
                    fig_height=8,
                    df=self.combo_df,
                    group_cols=["defenseFormation", "field_zone"],
                    value_col="passResult",
                    category_label="defenseFormation"
                )

                st.markdown(
                    """
                    #### Visualization Summary
                    - The **Heatmap** provides a high-level view of the effectiveness of defensive formations across different field zones and pass outcomes.
                    - The **Circular Bar Plot** adds context by showing which formations were most frequently used by the defensive teams in various field zones.

                    **How to Use It:**
                    - Start with the heatmap to identify where defensive formations were most effective (e.g., dark colors near "Incomplete" or "Intercepted").
                    - Use the circular bar plot to explore the frequency of each formation in different field zones and correlate it with the heatmap trends.

                    **Takeaway:**
                    This visualization provides a clear breakdown of how defensive formations were employed and their impact on pass outcomes across key field zones. It helps answer:
                    - Which defensive formations are most effective in specific field zones?
                    - Where do defensive formations succeed or fail against offensive plays?
                    - How can you adapt offensive strategies to counter specific defensive formations?

                    By combining the heatmap with the circular bar plot, we can analyze defensive setups more effectively and adjust strategies accordingly.
                    """
                )

                st.divider()
                st.markdown("<h3 style='text-align: center; color: gray;'>Receiver Alignments and Field Zones</h3>", unsafe_allow_html=True)
                st.markdown(
                    """
                    This visualization combines two types of charts—a **Heatmap** and a **Circular Bar Plot**—to analyze how offenses set up their receiver alignments (`receiverAlignment`) 
                    in various field zones. The goal is to understand how these alignments impacted pass outcomes (`passResult`), such as completions, incompletions, sacks, 
                    interceptions, or scrambles, across four key field zones (Opponent Red Zone, Midfield, Own Territory, Own Deep Zone).
                    """
                )
                receiver_alignmentn_heatmap_text , receiver_alignmentn_plot_text = st.columns(2)

                with receiver_alignmentn_heatmap_text:
                    st.markdown(
                        """
                        #### Heatmap
                        **What It Shows:**
                        - The heatmap shows the proportion of pass outcomes (`passResult`) for different combinations of field zones and receiver alignments.
                        - **X-axis:** Pass outcomes (`passResult`) such as Complete, Incomplete, Sack, Intercepted.
                        - **Y-axis:** Receiver alignments (`receiverAlignment`), such as 3x1, 2x2, or 1x1.
                        - **Darker colors:** Indicate where certain outcomes are more frequent:
                        - Darker near **Complete:** Suggests the receiver alignment was effective in producing completions in the corresponding field zone.
                        - Darker near **Incomplete** or **Intercepted:** Suggests the alignment was less effective, or the defense disrupted the play.
                        - Lighter or transparent colors indicate negligible or less frequent outcomes.

                        **Why It Matters:**
                        - Highlights where offenses excel or struggle based on their receiver alignments in specific field zones.
                        - Identifies which alignments lead to successful or unsuccessful pass outcomes.

                        **Key Insight:**
                        - This heatmap helps pinpoint receiver alignments that worked best in specific field zones.
                        - For example, a darker shade near "**Complete**" suggests a 3x2 alignment was effective in the **Midfield**.
                        - Conversely, a darker shade near "**Incomplete**" indicates where the alignment failed to produce successful plays.

                        """
                    )

                with receiver_alignmentn_plot_text:
                    st.markdown(
                        """
                        #### Circular Bar Plot (Radial Bar Chart)
                        **What It Shows:**
                        - The circular chart is divided into four sections, each representing a field zone:
                        - **Opponent Red Zone** (0-25 yards)
                        - **Midfield** (26-50 yards)
                        - **Own Territory** (51-75 yards)
                        - **Own Deep Zone** (76-100 yards).
                        - Each bar represents a receiver alignment (`receiverAlignment`) used in that field zone (e.g., 3x1, 2x2, 3x2).
                        - The length of the bar shows how frequently that alignment was used in the corresponding field zone.

                        **Why It Matters:**
                        - **Peak of the Bar:** Longer bars highlight alignments that were frequently used in specific field zones.
                        - **Color:** Colors differentiate field zones, making it easier to compare how offenses set up their receivers in different areas of the field.

                        **Key Insight:**
                        - Use this chart to identify field zones where specific receiver alignments were heavily employed.
                        - For example, the chart might show that 2x1 alignments are frequently used in the **Opponent Red Zone**, while 3x2 alignments are more common in the **Midfield**.

                        """
                    )

                self.plot_combined(
                    fig_width=0.5,
                    fig_height=0.5,
                    df=self.combo_df,
                    group_cols=["receiverAlignment", "field_zone"],
                    value_col="passResult",
                    category_label="receiverAlignment",
                    inner_radius=80
                )
                st.markdown(
                    """
                    #### Visualization Summary
                    - The **Heatmap** provides a high-level view of the effectiveness of receiver alignments across different field zones and pass outcomes.
                    - The **Circular Bar Plot** adds context by showing which alignments were most frequently used in various field zones.

                    **How to Use It:**
                    - Start with the heatmap to identify where receiver alignments were most effective (e.g., dark colors near "Complete" or "Incomplete").
                    - Use the circular bar plot to explore the frequency of each alignment in different field zones and correlate it with the heatmap trends.

                    **Takeaway:**
                    This visualization provides a clear breakdown of how offensive receiver alignments were employed and their impact on pass outcomes across key field zones. It helps answer:
                    - Which receiver alignments are most effective in specific field zones?
                    - Where do alignments succeed or fail based on pass outcomes?
                    - How can you adjust receiver alignments to maximize success in different field zones?

                    By combining the heatmap with the circular bar plot, we can analyze offensive setups more effectively and adapt strategies to counter defensive alignments.
                    """
                )
                st.divider()
                st.markdown("<h3 style='text-align: center; color: gray;'>Offense Formation and Field Zones</h3>", unsafe_allow_html=True)
                st.markdown(
                    """
                    This visualization combines two types of charts—a **Heatmap** and a **Circular Bar Plot**—to analyze how offenses employed different formations (`offenseFormation`) across 
                    various field zones. The goal is to understand how these formations impacted pass outcomes (`passResult`), such as completions, incompletions, sacks, interceptions, or scrambles, 
                    across four key field zones (Opponent Red Zone, Midfield, Own Territory, Own Deep Zone).

                    """
                )
                offense_formation_heatmap_text , offense_formation_plot_text = st.columns(2)

                with offense_formation_heatmap_text:
                    st.markdown(
                        """
                        #### Heatmap
                        **What It Shows:**
                        - The heatmap displays the proportion of pass outcomes (`passResult`) for different combinations of field zones and offensive formations.
                        - **X-axis:** Pass outcomes (`passResult`) such as Complete, Incomplete, Sack, Intercepted.
                        - **Y-axis:** Offensive formations (`offenseFormation`), such as Shotgun, Singleback, Empty, Jumbo, or I-Form.
                        - **Darker colors:** Indicate where certain outcomes are more frequent:
                        - Darker near **Complete:** Suggests the offensive formation was effective in the corresponding field zone.
                        - Darker near **Incomplete** or **Intercepted:** Suggests the formation struggled in that zone.
                        - Lighter or transparent shades indicate less frequent or less impactful combinations of outcomes and formations.

                        **Why It Matters:**
                        - Highlights where offensive formations excel or falter in specific field zones.
                        - Identifies which formations contribute to successful or unsuccessful pass outcomes.

                        **Key Insight:**
                        - This heatmap helps pinpoint offensive formations that worked best in specific field zones.
                        - For example, a darker shade near "**Complete**" suggests that the Shotgun formation was effective in a particular zone.
                        - A darker shade near "**Incomplete**" or "**Intercepted**" indicates where the formation may have been less effective.

                        """
                    )
                
                with offense_formation_plot_text:
                    st.markdown(
                        """
                        #### Circular Bar Plot (Radial Bar Chart)
                        **What It Shows:**
                        - The circular chart is divided into four sections, each representing a field zone:
                        - **Opponent Red Zone** (0-25 yards)
                        - **Midfield** (26-50 yards)
                        - **Own Territory** (51-75 yards)
                        - **Own Deep Zone** (76-100 yards).
                        - Each bar represents an offensive formation (`offenseFormation`) used in that field zone (e.g., Shotgun, Singleback).
                        - The length of the bar shows how frequently that formation was used in the corresponding field zone.

                        **Why It Matters:**
                        - **Peak of the Bar:** Longer bars highlight formations that were frequently used in specific field zones.
                        - **Color:** Colors differentiate field zones, making it easier to compare how offenses employed formations across the field.

                        **Key Insight:**
                        - Use this chart to identify field zones where specific formations were heavily employed.
                        - For example, the chart might show that the Shotgun formation is frequently used, while Jumbo is less common overall*.

                        """
                    )

                self.plot_combined(
                    fig_width=0.5,
                    fig_height=0.5,
                    df=self.combo_df,
                    group_cols=["offenseFormation", "field_zone"],
                    value_col="passResult",
                    category_label="offenseFormation"
                )

                st.markdown(
                    """
                    #### Visualization Summary
                    - The **Heatmap** provides a high-level view of the effectiveness of offensive formations across different field zones and pass outcomes.
                    - The **Circular Bar Plot** adds context by showing which formations were most frequently used in various field zones.

                    **How to Use It:**
                    - Start with the heatmap to identify where offensive formations were most effective (e.g., dark colors near "Complete" or "Incomplete").
                    - Use the circular bar plot to explore the frequency of each formation in different field zones and correlate it with the heatmap trends.

                    **Takeaway:**
                    This visualization provides a clear breakdown of how offensive formations were employed and their impact on pass outcomes across key field zones. It helps answer:
                    - Which offensive formations are most effective in specific field zones?
                    - Where do formations succeed or fail based on pass outcomes?
                    - How can you adjust offensive formations to maximize success in different field zones?

                    By combining the heatmap with the circular bar plot, we can analyze offensive setups more effectively and optimize strategies for each field zone.
                    """
                )

        

predictor = NFLAdvancedPlaygroundSimulator(model_directory, player_data, historical_data, logo_folder_path, offense_player_path)
predictor.run_streamlit_app()
