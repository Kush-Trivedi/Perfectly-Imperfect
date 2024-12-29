import streamlit as st

st.title(":violet[Perfectly-Imperfect]: Pre-Snap Mastery with Dink & Dunk, Game Play Simulation, and QB’s Radar")

st.markdown(
    """
    #### Introduction
    Please allow me to introduce myself I am *Kush Trivedi a Sr.ML/Data Scientist* my journey in analyzing NFL Big Data Bowl over the past few years has taught me a valuable lesson.
    Despite my technical expertise & enthusiasm towards football, I always lacked the crucial domain knowledge of football strategies and their execution on the field. 
    Moreover, this year, I am very thankful to **DFO** ([**DeMarkco Butler**](https://ucmathletics.com/staff-directory/demarkco-butler/388)) for all his support throughout this project & 
    I truly appreciate the live game valuable insights from [**Coach Josh Lenberson’s**](https://themiaa.com/news/2023/12/6/football-josh-lamberson-named-afca-division-ii-super-region-coach-of-the-year.aspx), who is the ***AFCA Super Region II Coach of the Year 2023*** and the ***2023 MIAA Coach of the Year***.

    As NFL offenses continue to innovate in the face of aggressive defensive schemes, short-passing plays have emerged as the cornerstone of modern strategy. 
    With defenses emphasizing pressure and coverage schemes to limit deep balls, offenses are responding by capitalizing on quick, efficient passes **DINKING** & **DUNKING** their way to dominance.

    **The numbers back it up**: Short passes, particularly to the *middle*, *right*, and *left* have consistently shown higher success rates in sustaining drives and avoiding costly turnovers. 
    Teams that master these plays gain an edge in controlling the *clock*, converting key *downs*, and maximizing *Expected Points Added (EPA)* per play. 

    **The numbers back it up! → For real? → Yes**
    
    - *But how do offensive play tendencies achieve this precision based on pre-snap configurations?*
    - *How can we predict yard gains and determine the most effective offensive strategies in different scenarios?*
    - *Which receiver should the quarterback target based on pre-snap formations?*
    - *How pre-snap formations and tendencies could predict post-snap outcomes?*
   
    This project provides answers to such questions by visualizes the *trends*, analyzing *quaterbacks-receivers patterns*, *strategies*, hands on with ***real time game play scenario simulator*** & revealing not only what happens but also why.
    
    #### Playground
    """
)

col_1, col_2, col_3 = st.columns(3)

with col_1:
    with st.container(border=True, height=320):
        st.markdown(
            """
            :violet[**Dink & Dunk Report**]

            A handy report that gives the ***historical trends***, ***tendencies***, & ***routes-pass patterns*** for 32 NFL Teams. 
            
            It mainly focuses on short-pass strategies, known as "**Dink & Dunk**" with stats for short and deep passes we can also drill down to individual team level stats with a detailed breakdown of short passes.
            """
        )

with col_2:
    with st.container(border=True, height=320):
        st.markdown(
            """
            :violet[**Game Play Simulator**]

            A real time game play simulator where we can provide custom game play scenarios and get **88%-93%** accurate decisions on player **performance**, **predicted yard gain outcomes**, **predicted offensive strategies** & much more.

            Every insight is tailored using tracking data to empower coaches with data-driven decision.
            """
        )


with col_3:
    with st.container(border=True, height=320):
        st.markdown(
            """
            :violet[**Quarterback's Radar**]

            This model help quarterbacks **identify the best receiver** for a  **Deep Passes** play by using tracking data with **video animation**. 
            
            It uses two advanced heuristic search algorithms "**Beam Search**" & "**Progressive Widening Search**". Both are designed to identify the "Best Receiver" before the snap happens.
            """
        )


st.markdown("""

    #### Methodology
    Dink & Dunk Report, Gameplay Simulator, and Quarterback’s Radar by extensively engineering and transforming raw tracking data to extract meaningful insights. For the Dink & Dunk Report, we identified pass plays and categorized them into short-pass strategies by analyzing formations, down and distance, player routes, and defensive coverage, despite the tracking data not providing explicit details. 
    
    The Gameplay Simulator was created by defining game situations like quarter, down, and yards to go, then analyzing offensive and defensive player data to train machine learning models that predict yard gains and strategies with high accuracy. 
            
    For Quarterback’s Radar, we tracked receiver positions and performances, then applied advanced algorithms to determine the best receiver for deep passes before the snap, integrating this with video animations. Across all tools, we built data processing pipelines to classify plays, transform and clean the data, and develop models that provide coaches and quarterbacks with reliable, data-driven insights.
""")

col_4, col_5, col_6 = st.columns(3)

with col_4:
    with st.container(border=True, height=980):
        st.markdown(
            """
            :violet[**Dink & Dunk**] **Pipeline**

            Developing the Dink & Dunk Report required extensive data engineering to analyze short-pass strategies for all **32 NFL teams**. We began by identifying pass plays from raw tracking data, distinguishing between short and deep passes. 
            
            1. For short passes, we categorized them into types such as **Singleback Balanced Short Pass**, **Short Middle, Right, & Left Pass**, **Red Zone Short Pass**, **Short Yardage Pass**, **Midfield Short Pass** & **MANY OTHER such pass**. 
            
            2. This classification was based on factors like ***receiver alignment, offense formation, quater, down and distance, yards to go, play direction, player routes, and defensive coverage*** etc. 
            
            3. Since the tracking data did not provide explicit details that require for Dink & Dunk, we transformed and categorized the data by analyzing route patterns, first downs, EPA's and other relevant metrics and stored the processed data. 
            
            4. This processed data allows users to drill down into individual team statistics with a detailed breakdown of short-pass strategies.
            """
        )


with col_5:
    with st.container(border=True, height=980):
        st.markdown(
            """
            :violet[**Gameplay Simulator**] **Pipeline**

            The Gameplay Simulator was built to provide real-time predictions for custom game scenarios with **88%-93%** accuracy:

            1. **Game Situation Identification**: We extracted game situations including quarter, down, yards to go, yards to the end zone, game half, and time remaining.
            2. **Offensive and Defensive Data Analysis**: For the offense, we identified all 11 players, their routes, positions, formations, play types (pass or run), scores, winning probabilities, and individual performances. For the defense, we analyzed the 11 defensive players, their coverage types (zone or man), positions, formations, play types, scores, winning probabilities, and individual performances.
            3. **Data Transformation and Cleaning**: Since the tracking data did not provide all necessary details, we transformed and cleaned the data to create a comprehensive dataset suitable for machine learning models.
            4. **Model Training and Prediction**: Using the processed data, we trained machine learning models to predict yard gain outcomes and offensive strategies. This simulator empowers coaches with data-driven insights to make informed decisions during games.

            """
        )


with col_6:
    with st.container(border=True,  height=980):
        st.markdown(
            """
            :violet[**Quarterback’s Radar**] **Pipeline**

            Quarterback’s Radar assists us in selecting the best receiver for deep pass plays by leveraging tracking data and video animations:

            1. **Algorithm Development**: We implemented two advanced heuristic search algorithms, Beam Search and Progressive Widening Search, to evaluate receiver options in real-time.

            2. **Players Tracking**: We tracked the XY locations and performance metrics, calulated distance for individual offense-defense players in regards to QB.
            
            3. **Pre-Snap Analysis**: The algorithms analyze the tracking data before the snap to identify the optimal receiver based on factors such as receiver positioning, defensive coverage, and thier current postion on filed.
            
            4. **Integration with Video Animation**: The selected receiver is highlighted through video animations, providing us with clear, actionable insights. Since the tracking data did not explicitly indicate the best receiver, we developed these algorithms to interpret the data and make accurate receiver recommendations.
            """
        )


