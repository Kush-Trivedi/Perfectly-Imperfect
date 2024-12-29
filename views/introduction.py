import streamlit as st

st.title(":violet[Perfectly-Imperfect]: Pre-Snap Mastery with Dink & Dunk, Game Play Simulation, and QB’s Radar")

st.markdown(
    """
    #### Introduction
    Please allow me to introduce myself I am *Kush Trivedi a Sr.ML/Data Scientist* my journey in analyzing NFL Big Data Bowl over the past few years has taught me a valuable lesson.
    Despite my technical expertise & enthusiasm towards football, I always lacked the crucial domain knowledge of football strategies and their execution on the field. 
    Moreover, this year, I am very thankful to **DFO** ([**DeMarkco Butler**](https://ucmathletics.com/staff-directory/demarkco-butler/388)) for all his support throughout this project & 
    I truly appreciate the live game valuable insights from [**Coach Josh Lenberson’s**](https://themiaa.com/news/2023/12/6/football-josh-lamberson-named-afca-division-ii-super-region-coach-of-the-year.aspx), who is the ***AFCA Super Region III Coach of the Year 2023*** and the ***2023 MIAA Coach of the Year***.

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


