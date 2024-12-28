import streamlit as st

st.title(":violet[NFL] Big Data Bowl 2025")

st.markdown(
    """
    #### Introduction
    Please allow me to intoduce myself I am *Kush Trivedi a Sr.ML/Data Scientist* my journey in analyzing NFL Big Data Bowl over the past few years has taught me a valuable lesson.
    Despite my technical expertise & enthusiasm towards football, I always lacked the crucial domain knowledge of football strategies and their execution on the field. 
    Moreover, this year, I am very thankful to **DFO** ([**DeMarkco Butler**](https://ucmathletics.com/staff-directory/demarkco-butler/388)) for all his support throughout this project & 
    I truly appreciate the live game valuable insights from [**Coach Josh Lenberson’s**](https://themiaa.com/news/2023/12/6/football-josh-lamberson-named-afca-division-ii-super-region-coach-of-the-year.aspx), who is the ***AFCA Super Region III Coach of the Year 2023*** and the ***2023 MIAA Coach of the Year***.

    As NFL offenses continue to innovate in the face of aggressive defensive schemes, short-passing plays have emerged as the cornerstone of modern strategy. 
    With defenses emphasizing pressure and coverage schemes to limit deep balls, offenses are responding by capitalizing on quick, efficient **Passes—DINKING** and **DUNKING** their way to dominance.

    **The numbers back it up**: Short passes, particularly to the *middle*, *left*, and *right* have consistently shown higher success rates in sustaining drives and avoiding costly turnovers. 
    Teams that master these plays gain an edge in controlling the *clock*, converting key *downs*, and maximizing *Expected Points Added (EPA)* per play. **The numbers back it up! → For real? → Yes**

    
    - But how do teams achieve this precision? 
    - What drives the success of short passes in different scenarios? 
    - How pre-snap formations and tendencies could predict post-snap outcomes?
   
    This project provides answers to such questions by visualizes the *trends*, analyzing *quaterbacks-receivers patterns*, *strategies*, hands on with ***real time game play scenario simulator*** & revealing not only what happens but also why.
    
    #### Playground
    """
)

col_1, col_2, col_3 = st.columns(3)

with col_1:
    with st.container(border=True, height=200):
        st.markdown(
            """
            **Dink & Dunk Report**:

            A handy report that gives the ***historical trends***, ***tendencies***, & ***routes-pass patterns*** for 32 NFL Teams.
            """
        )

with col_2:
    with st.container(border=True, height=200):
        st.markdown(
            """
            **Game Play Simulator**:

            A real time game play simulator where you can provide any game play scenarios and get **88%-93%** accurate predicted data-driven decisions on player performance, predicted yard gain outcomes, predicted offensive strategies & much more.
            """
        )


with col_3:
    with st.container(border=True, height=200):
        st.markdown(
            """
            **QB's Radar**:

            This model help quarterbacks **identify the best receiver** for a  **Deep Passes** play by using tracking data with **video animation**.
            """
        )


