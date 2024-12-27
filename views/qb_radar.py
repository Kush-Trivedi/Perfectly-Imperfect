import os
import time
import streamlit as st
from pathlib import Path

st.title("The :orange[QB's] - :blue[Receiver] Radar")

team_names = [
    'KC', 'MIA', 'WAS'
]

logo_folder = Path("assets/game-play")

what, why = st.columns(2)
with what:
    with st.container(border=True):
        st.markdown(
            """
            **What is QB's Radar**? \n
            QB's Radar is a model designed to help quarterbacks identify the best receiver for a play. It uses two advanced methods:
            
            1. **Strategic Receiver**: This is the receiver predicted by a technique called **Progressive Widening Search**.
            2. **Primary Receiver**: This is the receiver predicted by a different method called **Beam Target Search**.
            3. **Best Receiver**: When both models agree on the receiver, and it matches *most of the actual game choice*.

            The model’s insights are especially valuable for **Deep Passes**, where decision-making is critical.
            """
        )


with why:
    with st.container(border=True):
        st.markdown(
            """
            **Why QB's Radar Matters**:

            By using QB's Radar, quarterbacks can improve offensive efficiency and yardage on deep plays.
            
            1. **Improves Decision-Making**: Helps quarterbacks avoid risky plays and identify receivers with better potential.
            2. **Highlights Missed Opportunities**: Shows how passing to the predicted receiver could have changed the game.
            3. **Deep Pass Excellence**: Excels at identifying the best options for deep passes, optimizing offensive strategy.

            With QB's Radar, quarterbacks can improve offensive efficiency, reduce turnovers, and make smarter choices to maximize yardage and 
            game impact.
            """
        )

st.divider()
st.markdown("<h3 style='color: gray;'>QB-Receiver Insights: Select Team, Game, and Play</h3>", unsafe_allow_html=True)

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
    
    st.divider()

    if os.path.exists(image_path):
        st.image(image_path, caption="Receiver Scores", use_container_width=True)
    else:
        st.warning("Receiver scores image not found for the selected Game ID and Play ID.")

    st.divider()

    st.markdown("<h3 style='color: gray;'>Technical Findings and Results</h3>", unsafe_allow_html=True)

    st.markdown(
        """    
        The **QB's Radar**, is designed to predict the best receiver for a quarterback to target. It uses two predictive approaches:

        1. **Strategic Receiver**: Predicted using the Progressive Widening Search algorithm, optimized for safer and potentially higher-value plays.
        2. **Primary Receiver**: Predicted using the Beam Search algorithm, which often aligns with conventional quarterback decisions.
        3. **Best Receiver**: Occurs when both models predict the same receiver and the predicted receiver matches the actual receiver targeted during the game most of the time.

        This report provides a detailed evaluation focusing on its ability to identify superior receiver options, particularly for deep passes, 
        using a variety of statistical and machine learning methods.
        """
    )

    test_1 , test_2 = st.columns(2)

    with test_1:
        with st.container(border=True,height=600):
            st.markdown(
                """
                1. **Paired t-Test and Wilcoxon Test for Yards Gained**
                    - **Objective**: Compare actual yards gained (`yardsGained`) with the predicted distance to the Strategic Receiver (`progressive_widening_search_receiver_distance`).
                    - **Results**: 
                        - **Paired t-test**:
                            - **T-statistic**: 14.783
                            - **P-value**: 4.17 × 10⁻³⁹
                            - **Conclusion**: The predicted receivers significantly outperformed the actual receivers in terms of yards gained.
                        - **Wilcoxon Test**:
                            - **Wilcoxon Statistic**: 7489.0
                            - **P-value**: 3.49 × 10⁻³⁸
                        - **Conclusion**: The non-parametric Wilcoxon test confirms the significant difference observed in the t-test, indicating robust results.

                """
            )

    with test_2:
        with st.container(border=True,height=600):
            st.markdown(
                """
                2. **Paired t-Test and Wilcoxon Test for Score Comparison**
                    - **Objective**: Compare actual yards gained (`yardsGained`) with the model-predicted scores (`progressive_widening_search_widening_receiver_score`).
                    - **Results**:
                        - **Paired t-test**:
                            - **T-statistic**: -37.882
                            - **P-value**: 1.06 × 10⁻¹²⁸
                            - **Conclusion**: The predicted scores show a statistically significant difference, confirming that the Strategic Receiver is often better positioned for high-value plays.
                        - **Wilcoxon Test**:
                            - **Wilcoxon Statistic**: 0.0
                            - **P-value**: 6.75 × 10⁻⁶²
                        - **Conclusion**: The Wilcoxon test strongly supports the t-test result, highlighting the consistent superiority of predicted scores.
                """

            )

    st.divider()

    test_3, test_4 = st.columns(2)

    with test_3:
        with st.container(border=True, height=400):
            st.markdown(
                """
                3. **Bootstrap Analysis**:
                - **Objective**: Quantify the mean difference in yards gained between actual receivers and predicted Strategic Receivers and compute a confidence interval.
                - **Results**:
                    - **Bootstrap Mean Difference**: 10.21 yards.
                    - **95% Confidence Interval**: [8.69, 11.77]
                - **Conclusion**: The predicted receivers gained significantly more yards than the actual receivers. The confidence interval does not include `0`, providing strong evidence for the model’s effectiveness.
                """
            )
    
    with test_4:
        with st.container(border=True, height=400):
            st.markdown(
                """
                4. **Effect Size (Cohen's d)**:
                - **Objective**: Measure the practical significance of the difference between actual yards gained and predicted receiver performance.
                - **Results**:
                    - **Cohen's d**: 0.892
                - **Conclusion**: This large effect size indicates a strong practical significance, confirming that the Strategic Receivers provide a substantial advantage in terms of yardage.
                """
            )

    st.divider()
    test_5, test_6 = st.columns(2)

    with test_5:
        with st.container(border=True, height=400):
            st.markdown(
                """
                5. **Analysis of Variance (ANOVA)**:
                - **Objective**: Evaluate whether pass location (left, middle, right) influences the effectiveness of yards gained.
                - **Results**:
                    - **F-statistic**: Computed internally.
                    - **P-value**: 0.795
                - **Conclusion**: Pass location does not have a statistically significant effect on the effectiveness of the model. The predictions are consistent regardless of whether the pass is to the left, middle, or right.
                
                """
            )
    
    with test_6:
        with st.container(border=True, height=400):
            st.markdown(
                """
                6. **Chi-Square Test for Pass Location and Success**:
                - **Objective**: Assess whether pass location influences the likelihood of the Strategic Receiver outperforming the actual receiver.
                - **Results**:
                    - **Chi-Square Statistic**: Computed internally.
                    - **P-value**: 0.906
                - **Conclusion**: There is no statistically significant association between pass location and the success of passes to the Strategic Receiver.

                """
            )

    st.divider()

    st.markdown(
        """
        #### :violet[Missed Opportunities] and Use Cases

        1. **Incomplete Passes**:
            - In several plays, the model predicted a **Strategic Receiver**, but the quarterback targeted a different receiver, resulting in an **incomplete pass**.
            - **Potential Outcome**: If the QB had passed to the predicted receiver, the outcome could have been a completion with better yardage.

        2. **Safe vs. Risky Choices**:
            - The model often identifies a **safer receiver** (Strategic Receiver) who might gain fewer yards but minimizes risk (e.g., avoiding interceptions or turnovers).
            - **Example**: The QB passed to the Primary Receiver, resulting in a completion but with increased risk. The Strategic Receiver would have been a safer option.

        3. **Game-Changing Potential**:
            - In some cases, the model’s Strategic Receiver prediction identified receivers with significantly higher yardage potential than the Primary Receiver.
            - **Example**: Passing to the Strategic Receiver could have resulted in a significant yardage gain and potentially altered the game outcome.
        
        """
    )
    st.divider()
    st.markdown(
        """
        #### :green[Conclusion]

        The **QB's Radar** demonstrates its ability to significantly enhance quarterback decision-making, particularly for **deep passes**. By leveraging statistical tests and real-world scenarios, the analysis validates the model’s predictive power:

        - **Statistically significant improvement** in yards gained and score predictions.
        - **Large effect size** confirming practical advantages.
        - **Missed opportunities** highlight areas where the model could have changed game outcomes.

        With further refinement, **QB's Radar** has the potential to optimize decision-making across all pass types, making it a valuable tool for NFL teams.

        """
    )

