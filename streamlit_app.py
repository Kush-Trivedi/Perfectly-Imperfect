import streamlit as st 

st.set_page_config(layout="wide")

st.markdown(
    r"""
    <style>
    .reportview-container {
            margin-top: -3em;
        }
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    r"""
    <style>
    div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
        height: 11rem;
        width: auto;
        display: block;  
        margin-left: auto;  
        margin-right: auto; 
    }

    div[data-testid="stSidebarHeader"], div[data-testid="stSidebarHeader"] > *,
    div[data-testid="collapsedControl"], div[data-testid="collapsedControl"] > * {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    [data-testid="stSidebarNav"]::before {
        content: "Perfectly Imperfect";
        margin-left: 20px;
        font-size: 30px;
        font-weight: bold;
        position: relative;
    }
    </style>
    """, unsafe_allow_html=True
)

# Introduction Page
introduction_page = st.Page(
    page="views/introduction.py",
    title="NFL Big Data Bowl 2025",
    icon=":material/demography:",  
    default=True
)

# Advance Game Predictor Page
game_play_simulator = st.Page(
    page="views/game_play_simulator.py",
    title="Game Play Simulator",
    icon=":material/sports_football:"
)

# Game Play Page
qb_passing_guide_page = st.Page(
    page="views/qb_radar.py",
    title="Quarterback's Radar",
    icon=":material/target:"
)

# Dink & Dunk Page
dink_dunck = st.Page(
    page="views/dink_dunk.py",
    title="Dink & Dunk Report",
    icon=":material/inventory:"
)



# Navigation Setup
pg = st.navigation(
    {
        "Intoduction": [introduction_page],
        "Playground": [game_play_simulator, dink_dunck, qb_passing_guide_page]
    }
)

st.logo("assets/navbar/logo.png")

# Run Navigation
pg.run()
