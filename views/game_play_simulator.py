
import os
import json
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


model_directory ="assets/models/yards_gained"
offense_model_dir = "assets/models/offense_strategy"
player_data = "assets/data/all_player_ratings.csv"
historical_data = "assets/data/season_2022_historical_data.csv"
logo_folder_path = "assets/logo"

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
    epsilon = 1e-6
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

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
def _load_historical_data(historical_data_path):
    try:
        return pd.read_csv(historical_data_path)
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.DataFrame()
    
@st.cache_data
def load_ratings(model_dir, quarter, down):
    try:
        file_path = os.path.join(model_dir, f"/Users/kushtrivedi/Desktop/nfl-big-data-bowl-2025/assets/player_ratings/all_{quarter}_{down}_player_rating.csv")
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found for Quarter {quarter} and Down {down}.")
        return pd.DataFrame()

@st.cache_data
def get_weighted_player_rating(player_df, player_names, quarter, down):
    overall_df = player_df
    specific_df = load_ratings(model_directory,quarter, down)

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
    def __init__(self, model_dir, player_data_path, historical_data_path, logo_folder_path):
        self.model_dir = model_dir
        self.player_data_path = player_data_path
        self.historical_data_path = historical_data_path

        self.loaded_model, self.loaded_encoder, self.loaded_scaler, self.loaded_y_labels = load_model_components(self.model_dir)
        self.player_df = _load_player_data(self.player_data_path)
        self.historical_data = _load_historical_data(self.historical_data_path)

        self._set_plot_defaults()
        self._initialize_categories()

        self.logo_folder = Path(logo_folder_path)
        self.team_names = self._load_team_names()
        self._team_full_names()


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
        """Load team names based on the logo file names in the folder."""
        if not self.logo_folder.exists():
            st.error(f"Logo folder not found: {self.logo_folder}")
            return []
        return [logo_file.stem for logo_file in self.logo_folder.glob("*.png")]

    def get_player_rating(self, player_names):
        """Retrieve player ratings based on player names."""
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
            quarter = st.selectbox("Quarter", options=[1, 2, 3, 4, 5], key="quarter")
            st.markdown(
                """
                    <style> div[data-baseweb="segmented-control"] { } div[data-baseweb="segmented-control"] button { } </style>
                """,
                unsafe_allow_html=True
            )
            option_map = {0: ":material/hourglass_top: ", 1: ":material/hourglass_bottom: ", 2: ":material/more_time: "}
            game_half = st.segmented_control( "Game Time Segment", options=option_map.keys(), format_func=lambda option: option_map[option], default=0 )
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
            down = st.selectbox("Down", options=[1, 2, 3, 4], key="down")
            quarter_seconds_remaining = st.slider("Seconds Remaining in a Quarter", min_value=0, max_value=900, value=420)
            minutes = quarter_seconds_remaining // 60
            seconds = quarter_seconds_remaining % 60
            st.markdown(f"Time left in Qtr: **{minutes:02}:{seconds:02}** (*MM:SS*)")
            
        with col3:
            yards_to_go = st.selectbox("Yards to Go", options=yards_to_go_options, key="yards_to_go")
            half_seconds_remaining = st.slider("Seconds Remaining in a Half", min_value=0, max_value=1800, value=1000)
            minutes = half_seconds_remaining // 60
            seconds = half_seconds_remaining % 60
            st.markdown(f"Half time left: **{minutes:02}:{seconds:02}** (*MM:SS*)")

        with col4:
            yards_to_endzone = st.selectbox("Yards to Endzone", options=yards_to_endzone_options, key="yards_endzone")
            game_seconds_remaining = st.slider("Seconds Remaining in the Game", min_value=0, max_value=3600, value=620)
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
                offense_team_name = st.selectbox("Team Name", options=self.team_names, key="offense_team_name")

            with offense_col_d:
                off_score = st.number_input("Score Points", key="offense_score", min_value=0, max_value=99, step=1, format="%d")

            with offense_col_e:
                off_timeout_remaining = st.selectbox("Timeout Left", options=[3, 2, 1])

            with offense_col_f:
                offense_wp = st.number_input("Win Probability", format="%.5f")

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

            offense_formation = st.selectbox("Offense Formation", options= unique_offense_formation)

            
            for i in range(11):
                col_player, col_position, col_route = st.columns(3)
                with col_player:
                    player = st.selectbox(f"Player {i + 1}", options=self.player_df['displayName'], key=f"offense_player_{i}")
                with col_position:
                    position = st.selectbox(f"Position", options=unique_offense_position, key=f"offense_position_{i}")
                    
                with col_route:
                    route = st.selectbox(f"Route", options=unique_routes, key=f"offense_route_{i}")
                    
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
            "DF", "PRE", "Unknown Assignment", "CFR", "Deep Run Support", 
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
                defense_play_selection = st.pills("Coverage", options=defesne_play_option_map.keys(), format_func=lambda option: defesne_play_option_map[option], selection_mode="single", default=0, key="defense_play_selection")
                defense_play_type = defesne_play_option_map[defense_play_selection]
                defense_play_type = defense_play_type.split(":")[-1].strip()

            defense_col_c, defense_col_d, defense_col_e, defense_col_f = st.columns(4)

            with defense_col_c:
                defense_team_name = st.selectbox("Team Name", options=self.team_names, key="defense_team_name")

            with defense_col_d:
                def_score = st.number_input("Score Points", key="defense_score", min_value=0, max_value=99, step=1, format="%d")

            with defense_col_e:
                def_timeout_remaining = st.selectbox("Timeout Left", options=[3,2,1],key="def_timeout")

            with defense_col_f:
                defense_wp = st.number_input("Win Probablity",key="defense_wp",format="%.5f")


            
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

                    
            defense_formation = st.selectbox("Defense Formation", options = unique_defense_formation)
            
            for i in range(11):
                def_col_player, def_col_position, col_coverage = st.columns(3)
                with def_col_player:
                    def_player = st.selectbox(f"Player {i + 1}", options=self.player_df['displayName'], key=f"defense_player_{i}")
                with def_col_position:
                    def_position = st.selectbox(f"Position", options=unique_defense_position, key=f"defense_position_{i}")
                    
                with col_coverage:
                    def_coverage = st.selectbox(f"Coverage", options=unique_coverage, key=f"defense_coverage_{i}")
                    
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


        clicked = st.button("Predict Game Play Yards", key="predict_button", use_container_width=True, type="primary")

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
    
                ### **Game Situational Context**
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

            st.markdown("<h1 style='text-align: center; color: black;'>Offense-Defense Tendencies</h1>", unsafe_allow_html=True)
            st.subheader("In-progress.....")
    


predictor = NFLAdvancedPlaygroundSimulator(model_directory, player_data, historical_data, logo_folder_path)
predictor.run_streamlit_app()
