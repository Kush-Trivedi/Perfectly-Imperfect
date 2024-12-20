import os
import time
import streamlit as st
from pathlib import Path

st.title("The :orange[QB's] - :blue[Receiver] Radar")

team_names = [
    'MIA', 'WAS', 'KC'
]

logo_folder = Path("assets/game-play")

selected_teams = st.multiselect("Select Teams:", team_names, default=["KC"])

@st.cache_data
def get_game_and_play_ids(selected_teams):
    game_play_dict = {}
    
    for team in selected_teams:
        team_folder_path = logo_folder / team

        if team_folder_path.exists():
            for game_folder in os.listdir(team_folder_path):
                game_path = os.path.join(team_folder_path, game_folder)
                if os.path.isdir(game_path) and game_folder.startswith("game_"):
                    game_id = game_folder.split("game_")[-1]
                    play_ids = []
                    
                    for play_folder in os.listdir(game_path):
                        if play_folder.startswith("play_"):
                            play_id = play_folder.split("play_")[-1]
                            play_ids.append(play_id)
                    
                    if play_ids:
                        game_play_dict[game_id] = {'team': team, 'play_ids': play_ids}
    
    return game_play_dict


game_play_dict = get_game_and_play_ids(selected_teams)

col1, col2 = st.columns(2)

with col1:
    selected_game_id = st.selectbox(
        "Select Game ID", 
        options=["Select a Game ID"] + list(game_play_dict.keys()), 
        index=0, 
        key="GameId"
    )

with col2:
    if selected_game_id != "Select a Game ID":
        play_options = ["Select a Play ID"] + game_play_dict[selected_game_id]['play_ids']
        selected_play_id = st.selectbox("Select Play ID", options=play_options, index=0, key="PlayId")
    else:
        selected_play_id = st.selectbox("Select Play ID", options=["Select a Play ID"], index=0, key="PlayId")
        st.warning("Please select a Game ID to see available plays.")


if selected_game_id != "Select a Game ID" and selected_play_id != "Select a Play ID":
    st.success(f"Selected Game ID: {selected_game_id}, Play ID: {selected_play_id}")

    team = game_play_dict[selected_game_id]['team']
    video_path = os.path.join(logo_folder, team, f"game_{selected_game_id}/play_{selected_play_id}", f"{selected_game_id}_{selected_play_id}_animation.mp4")
    
    image_path = os.path.join(logo_folder, team, f"game_{selected_game_id}/play_{selected_play_id}", f"{selected_game_id}_{selected_play_id}_receiver_scores.png")
    
    with st.spinner("We know it’s a sad part to wait... Your animation is in progress!"):
        progress_placeholder = st.empty() 
        progress_bar = progress_placeholder.progress(0)
        
        for percent in range(100):
            time.sleep(0.02) 
            progress_bar.progress(percent + 1)
        
        progress_placeholder.empty()
      
    if os.path.exists(video_path):
        st.video(video_path, muted=True)
    else:
        st.error("Video file not found for the selected Game ID and Play ID.")
    
    if os.path.exists(image_path):
        st.image(image_path, caption="Receiver Scores", use_container_width=True)
    else:
        st.warning("Receiver scores image not found for the selected Game ID and Play ID.")

