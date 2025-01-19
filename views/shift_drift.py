import base64
import pandas as pd
import streamlit as st
from pathlib import Path


st.title("The :green[Shift] & :blue[Drift] Report")


class MotionDataAnalysis:
    def __init__(self, file_path: str):
        self.motion_df = pd.read_csv(file_path)
        self.motion_df['motionCategory'] = self.motion_df['motionType'].apply(
            lambda x: 'No Motion' if x == 'No Motion' else 'Motion'
        )
        team_totals = self.motion_df.groupby('Team').size().rename('total_plays')
        self.motion_totals = (
            self.motion_df
            .groupby(['Team', 'motionCategory'])
            .size()
            .rename('motion_count')
            .reset_index()
            .merge(team_totals, on='Team')
        )
        self.motion_totals['percentage'] = (
            self.motion_totals['motion_count'] / self.motion_totals['total_plays'] * 100
        )

        filtered_motion_df = self.motion_df[self.motion_df['motionType'] != "No Motion"]

        self.my_df = (
            filtered_motion_df
            .groupby(["Team", "motionType", "motionDirection", "motionDisplayName"])
            .size()
            .rename('motion_player_direction_count')
            .reset_index()
        )
        self.my_df['motion_player_direction_percentage'] = self.my_df.groupby(
            ["Team", "motionType", "motionDirection"]
        )['motion_player_direction_count'].transform(
            lambda x: (x / x.sum() * 100).round(2)
        )

        self.my_df_2 = (
            filtered_motion_df
            .groupby(["Team", "motionType", "motionDirection"])
            .size()
            .rename('motion_direction_count')
            .reset_index()
        )
        self.my_df_2['motion_direction_percentage'] = self.my_df_2.groupby(
            ["Team", "motionType"]
        )['motion_direction_count'].transform(
            lambda x: (x / x.sum() * 100).round(2)
        )

        self.my_df_3 = (
            filtered_motion_df
            .groupby(["Team", "motionType"])
            .size()
            .rename('motion_type_count')
            .reset_index()
        )
        self.my_df_3['motion_type_percentage'] = self.my_df_3.groupby("Team")[
            'motion_type_count'
        ].transform(
            lambda x: (x / x.sum() * 100).round(2)
        )
    
    def generate_summary_dfs_by_play_type(self, team_name: str, play_type: str) -> dict:
        team_df = self.motion_df[
            (self.motion_df['Team'] == team_name) &
            (self.motion_df['play_type'].str.lower() == play_type.lower())
        ]
        if team_df.empty:
            return {"Error": f"No data found for {team_name} with play type '{play_type}'."}
        
        total_plays = len(team_df)
        summaries = {}

        cat_stats = team_df.groupby('motionCategory').agg(
            count=('motionCategory', 'size'),
            avg_yards=('yards_gained', 'mean'),
            median_epa=('epa', 'median')
        ).reset_index()

        cat_stats["avg_yards"] = cat_stats["avg_yards"].round(2)
        cat_stats['percentage'] = (cat_stats['count'] / total_plays * 100).round(2)
        cat_stats["Play Type"] = cat_stats.apply(
            lambda row: f"{row['motionCategory']}: {row['count']} ({row['percentage']}%)", axis=1
        )
        basic_stats = cat_stats[["Play Type", "avg_yards", "median_epa"]].rename(
            columns={"avg_yards": "Avg Yds", "median_epa": "EPA"}
        )
        summaries["BasicStats"] = basic_stats
        team_motion = team_df[team_df['motionType'] != "No Motion"]
        if not team_motion.empty:
            mt_stats = team_motion.groupby('motionType').agg(
                count=('motionType', 'size'),
                avg_yards=('yards_gained', 'mean'),
                median_epa=('epa', 'median')
            ).reset_index()
            mt_stats["avg_yards"] = mt_stats["avg_yards"].round(2)
            total_motion = mt_stats['count'].sum()
            mt_stats['motion_type_percentage'] = (mt_stats['count'] / total_motion * 100).round(2)
            mt_stats["Motion Type"] = mt_stats.apply(
                lambda row: f"{row['motionType']}: {row['motion_type_percentage']}%", axis=1
            )
            mt_df = mt_stats[["Motion Type", "avg_yards", "median_epa"]].rename(
                columns={"avg_yards": "Avg Yds", "median_epa": "EPA"}
            )
            summaries["MotionTypeBreakdown"] = mt_df
        else:
            summaries["MotionTypeBreakdown"] = pd.DataFrame({
                "Motion Type": ["No Motion plays available"],
                "Avg Yds": [None],
                "EPA": [None]
            })

        detailed_dict = {}
        if not team_motion.empty:
            md_stats = team_motion.groupby(['motionType', 'motionDirection']).agg(
                count=('motionDirection', 'size'),
                avg_yards=('yards_gained', 'mean'),
                median_epa=('epa', 'median')
            ).reset_index()
            md_stats["avg_yards"] = md_stats["avg_yards"].round(2)
            md_stats['motion_direction_percentage'] = md_stats.groupby('motionType')['count'].transform(
                lambda x: (x / x.sum() * 100).round(2)
            )
            for mtype, grp in md_stats.groupby('motionType'):
                grp = grp.copy()
                grp["Motion Movement"] = grp.apply(
                    lambda row: f"{row['motionDirection']}: {row['motion_direction_percentage']}%", axis=1
                )
                df_detail = grp[["Motion Movement", "avg_yards", "median_epa"]].rename(
                    columns={"avg_yards": "Avg Yds", "median_epa": "EPA"}
                )
                detailed_dict[mtype] = df_detail.reset_index(drop=True)
        summaries["DetailedMotionAnalysis"] = detailed_dict

        player_dict = {}
        if not team_motion.empty:
            p_stats = team_motion.groupby(['motionDisplayName', 'motionType', 'motionDirection']).agg(
                count=('motionDisplayName', 'size'),
                avg_yards=('yards_gained', 'mean'),
                median_epa=('epa', 'median')
            ).reset_index()
            p_stats["avg_yards"] = p_stats["avg_yards"].round(2)
            p_stats['motion_player_direction_percentage'] = p_stats.groupby(
                ['motionType', 'motionDirection']
            )['count'].transform(lambda x: (x / x.sum() * 100).round(2))
            for pname, grp in p_stats.groupby('motionDisplayName'):
                grp = grp.copy()
                grp["Player-Specific Tendencies"] = grp.apply(
                    lambda row: f"{row['motionType']} {row['motionDirection']}: {row['motion_player_direction_percentage']}%", axis=1
                )
                df_player = grp[["Player-Specific Tendencies", "avg_yards", "median_epa"]].rename(
                    columns={"avg_yards": "Avg Yds", "median_epa": "EPA"}
                )
                player_dict[pname] = df_player.reset_index(drop=True)
        summaries["PlayerSpecificTendencies"] = player_dict

        return summaries




file_path = Path("assets/data/all_teams_motion_analysis.csv")
analysis = MotionDataAnalysis(file_path)

logo_folder = Path("assets/logo")

if "team_order" not in st.session_state:
    st.session_state["team_order"] = {}


selected_teams = st.multiselect(
    "Select Teams:",
    options = [
            'LA', 'ATL', 'CAR', 'CHI', 
            'CIN', 'DET', 'HOU', 'MIA', 
            'NYJ', 'WAS', 'ARI', 'LAC', 
            'MIN', 'TEN', 'DAL', 'SEA', 
            'KC', 'BAL', 'CLE', 'JAX', 'NO', 
            'NYG', 'PIT', 'SF', 'DEN', 'LV', 
            'GB', 'BUF', 'PHI', 'IND', 'NE', 'TB'
        ] ,
    placeholder="Choose Teams",
    default=["KC"]
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
    
    for team in selected_teams:
        logo_path = logo_folder / f"{team}.png"

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
        pass_summaries = analysis.generate_summary_dfs_by_play_type(team, "pass")
        run_summaries = analysis.generate_summary_dfs_by_play_type(team, "run")

        st.markdown(container_style, unsafe_allow_html=True)

        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 3.5, 3])

            with col2:
                if logo_path and logo_path.exists():
                    st.image(str(logo_path))
                else:
                    st.write("No logo available.")

            col4, col5 =  st.columns(2)
            col6, col7 =  st.columns(2)
            col8, col9 =  st.columns(2)
            col10, col11 =  st.columns(2)
            with col4:
                st.markdown("<h3 style='text-align: center; color: black;'>Pass Plays </h3>", unsafe_allow_html=True)
                st.dataframe(pass_summaries["BasicStats"],use_container_width=True, hide_index=True)
            
            with col6:
                st.markdown("#### Motion Type Breakdown")
                st.dataframe(pass_summaries["MotionTypeBreakdown"],use_container_width=True, hide_index=True)
            
            with col8:
                st.markdown("#### Detailed Motion Analysis")
                for mtype, df in pass_summaries["DetailedMotionAnalysis"].items():
                    st.markdown(f"##### {mtype}")
                    st.dataframe(df,use_container_width=True, hide_index=True)
            with col10:
                st.markdown("#### Player-Specific Tendencies")
                for player, df in pass_summaries["PlayerSpecificTendencies"].items():
                    st.markdown(f"##### {player}")
                    st.dataframe(df,use_container_width=True, hide_index=True)

            with col5:
                st.markdown("<h3 style='text-align: center; color: black;'>Run Plays </h3>", unsafe_allow_html=True)
                st.dataframe(run_summaries["BasicStats"],use_container_width=True, hide_index=True)
            with col7:
                st.markdown("#### Motion Type Breakdown")
                st.dataframe(run_summaries["MotionTypeBreakdown"],use_container_width=True, hide_index=True)
            with col9:
                st.markdown("#### Detailed Motion Analysis")
                for mtype, df in run_summaries["DetailedMotionAnalysis"].items():
                    st.markdown(f"##### {mtype}")
                    st.dataframe(df,use_container_width=True, hide_index=True)
            with col11:
                st.markdown("#### Player-Specific Tendencies")
                for player, df in run_summaries["PlayerSpecificTendencies"].items():
                    st.markdown(f"##### {player}")
                    st.dataframe(df,use_container_width=True, hide_index=True)


