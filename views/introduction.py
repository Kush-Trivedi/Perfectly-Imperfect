import streamlit as st

st.title(":violet[Perfectly-Imperfect]: Pre-Snap Mastery with Dink & Dunk, QB’s Radar, and Scenario Game Play Simulation ")
st.divider()
st.markdown(
    """
    #### Introduction
    To begin with, I am very thankful to **DFO** ([**DeMarkco Butler**](https://ucmathletics.com/staff-directory/demarkco-butler/388)) for all his support throughout this project & 
    I truly appreciate the live game valuable insights from [**Coach Josh Lamberson's**](https://themiaa.com/news/2023/12/6/football-josh-lamberson-named-afca-division-ii-super-region-coach-of-the-year.aspx), who is the ***AFCA Super Region II Coach of the Year 2023*** and the ***2023 MIAA Coach of the Year***.

    **Objective 1**: Scouting players is arguably the most crucial task for NFL teams in their quest to acquire top talent. The 32 teams meticulously evaluate and compare players by stacking them against each other, such as ***"this player vs. that player"***. Traditionally, this process involves extensive analysis of player performance, character assessment, and data evaluation.
    **Bottom Line**: Make smarter decisions, faster—with data we can trust. While film study remains important, this scenario gameplay simulator is the icing on the cake, providing clear insights more quickly than traditional methods, showing you exactly how your football strategy will play out—all before we even step on the field.

    **Objective 2**: As NFL offenses continue to innovate in the face of aggressive defensive schemes, short-passing plays have emerged as the cornerstone of modern strategy. 
    With defenses emphasizing pressure and coverage schemes to limit deep balls, offenses are responding by capitalizing on quick, efficient passes **DINKING** & **DUNKING** their way to dominance. **The numbers back it up**: Short passes, particularly to the *middle*, *right*, and *left* have consistently shown higher success rates in sustaining drives and avoiding costly turnovers. 
    Teams that master these plays gain an edge in controlling the *clock*, converting key *downs*, and maximizing *Expected Points Added (EPA)* per play. 

    **Makes Smatter decisions! → The numbers back it up! → For real? → Yes**
    
    - *But can we use this for Scouting? How it will help for Scouting?*
    - *How pre-snap formations and tendencies could predict post-snap outcomes? What relation it has with Scouting?*
    - *How can we predict yard gains and determine the most effective offensive strategies in different scenarios? How can we leverage scouting for individual players?*
   
    This project provides answers to such questions in to **3 main playgrounds** by visualizes the *trends*, analyzing *quaterbacks-receivers patterns*, *strategies*, hands on with *real time game play scenario simulator* & revealing not only what happens but also why.
    
    """
)

st.divider()

col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.markdown(
        """
        :violet[**Game Play Simulator**]

        A real time game play simulator that helps you **create custom game scenarios**, **compare players side-by-side**, and **pick** the **best options across 32 teams**. It also predicts *yards gained*, and *offensive strategies* with over **80% accuracy** and provides trends & pattern on player - team performance.

        Every insight is tailored using tracking data to empower coaches with data-driven decision.
        """
    )

with col_2:
    st.markdown(
        """
        :violet[**Dink & Dunk Report**]

        A handy report that gives the ***historical trends***, ***player pass tendencies***, ***first down rates***, & ***routes-pass patterns*** for 32 NFL Teams, backed up with EPA to validate. 
        
        It mainly focuses on short-pass strategies, known as "**Dink & Dunk**" with stats for short and deep passes we can also drill down to individual team level stats with a detailed breakdown of short passes.
        """
    )

with col_3:
    st.markdown(
        """
        :violet[**Quarterback's Radar**]

        This model help quarterbacks **identify the best receiver** for a pass play by using tracking data with **video animation**.It uses two advanced heuristic search algorithms "**Beam Search**" & "**Progressive Widening Search**". Both are designed to identify the "Best Receiver" before the snap happens.
        """
    )


st.divider()


st.markdown("""
    #### Methodology
    The Gameplay Simulator requires extensively engineering and transforming of raw tracking data data to definie game situations like quarter, down, and yards to go, then analyzing offensive and defensive player data to train machine learning models that predict yard gains and strategies with high accuracy where we used **8 weeks data for training** and **tested on 9th week** and achieved above 80% accurate results.
       
    Dink & Dunk Report requires extensively engineering and transforming raw players-play data to extract meaningful insights. For the Dink & Dunk Report, we identified pass plays and categorized them into short & deep pass strategies by analyzing formations, down and distance, player routes, receiver alignment, and defensive coverage, despite the tracking data not providing explicit details. 
    
  
    For Quarterback’s Radar, we tracked receiver positions and distance, then applied advanced algorithms to determine the best receiver for passes before the snap, integrating this with a video animations where it highlights the Best Receiver and tracks the route ran by the receiver so we can look at the strategy indepth in future.
 
""")

st.divider()

col_4, col_5, col_6 = st.columns(3)

with col_4:
    st.markdown(
        """
        :violet[**Gameplay Simulator**] **Pipeline**

        The Gameplay Simulator was built to provide real-time predictions for custom game scenarios with **88%-93%** accuracy on test data (Week 9):

        1. **Game Situation Identification**: We extracted game situations including quarter, down, yards to go, yards to the end zone, game half, and time remaining.
        2. **Offensive and Defensive Data Analysis**: For the offense, we identified all 11 players, their routes, positions, formations, play types (pass or run), scores, winning probabilities, and individual performances. For the defense, we analyzed the 11 defensive players, their coverage types (zone or man), positions, formations, play types, scores, winning probabilities, and individual performances.
        3. **Data Transformation and Cleaning**: Since the tracking data did not provide all necessary details, we transformed and cleaned the data to create a comprehensive dataset suitable for machine learning models.
        4. **Model Training and Prediction**: Using the processed data, we trained machine learning models to predict yard gain outcomes and offensive strategies. This simulator empowers coaches with data-driven insights to make informed decisions during games.

        """
    )

with col_5:
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

with col_6:
    st.markdown(
        """
        :violet[**Quarterback’s Radar**] **Pipeline**

        Quarterback’s Radar assists us in selecting the best receiver for pass plays by leveraging tracking data and video animations:

        1. **Algorithm Development**: We implemented two advanced heuristic search algorithms, Beam Search and Progressive Widening Search, to evaluate receiver options in real-time.

        2. **Players Tracking**: We tracked the XY locations and performance metrics, calulated distance for individual offense-defense players in regards to QB.
        
        3. **Pre-Snap Analysis**: The algorithms analyze the tracking data before the snap to identify the optimal receiver based on factors such as receiver positioning, defensive coverage, and thier current postion on filed.
        
        4. **Integration with Video Animation**: The selected receiver is highlighted through video animations, providing us with clear, actionable insights. Since the tracking data did not explicitly indicate the best receiver, we developed these algorithms to interpret the data and make accurate receiver recommendations.
        """
    )

st.divider()


st.markdown(
    """

    #### How to Leverage for Scouting Purposes

    """
)

with st.container(border=True):
    st.markdown("**1. :violet[Gameplay Simulator] Guide**")
    st.markdown(
        """
        After running the Gameplay Simulator Pipeline methodology, focusing on player tracking and play details. In this process, we added new information that wasn't included in the tracking dataset, such as different defense 
        formations(***4-3 Defense, Big Nickel*** and ***many other***) and offensive strategies(***Short Middle, Left, RIght, Pistol Quick Pass, Shotgun Deep Attack***, & many more other strategies which can be found in code section).
        We also filled in gaps by adding routes for players who didn't have any listed, like ***Pass, Run, Lead, Kneel***, and ***Quick Block***, with help from a coach and friends who guided me on football terminology. 
        Similarly, we created new defensive coverages for players without assigned coverage assignment. Additionally, we incorporated extra data from the [NFLverse dataset](https://www.nflfastr.com/articles/field_descriptions.html) and performed feature 
        engineering to identify correlations, which led us to develop new features for better accuracy. Not only but also to evaluate player performance, we introduced a rating system on a scale of ***1 to 10***, assessing their effectiveness on each down, quarter, 
        and overall in the game. Finally, we leveraged this comprehensive data—including player ratings, offensive and defensive statistics, and game situations—to predict yards gained and potential offensive strategies. 
        
        """
    )

    left, middle, right = st.columns([1,8,1])
    with left:
        pass

    with middle:
        st.markdown(
            """
            **:violet[Guess]** the **:violet[Game Scenario]** Before we got to next page → ***How many Yards Offense Can gain? → What Strategy Offense will use?***
            """
        )

    with right:
        pass

    st.markdown(
        """
        **:orange[Scenario]**: The **Jets are on offense**, and they're in a challenging situation. They're trailing the Bills on the **scoreboard (3-14)** and have a long way to go to score (**65 yards to the end zone**). It's **Quarter 2, Down 2,** with **10 yards to go**, and the game is in the first half, with **4:24min remaining** in this half and a total game time left of 34:24min.

        """
    )

    f, s, t, fo = st.columns(4)

    with s:
        st.markdown("**:green[Offense ]**")

    with fo:
        st.markdown("**:blue[Defense]**")

    off, defe = st.columns(2)
    with off:
        st.markdown(
            """
            The Jets have **decided to pass the ball**, which makes sense since they need to move the ball quickly to catch up. Their quarterback, Zach Wilson, is in a **shotgun formation**, meaning he's standing a few steps behind the center when he gets the ball and given that they have **22.15% win probability**.
            """
        )

        st.markdown(
            """
            Here’s a look at the Jets' **offensive lineup**:
            - Duane Brown – Position: T (Tackle)
            - Cedric Ogbuehi – Position: T (Tackle)
            - Laken Tomlinson – Position: G (Guard)
            - Connor McGovern – Position: C (Center)
            - Tyler Conklin – Position: TE (Tight End)
            - Nate Herbig – Position: G (Guard)
            - Denzel Mims – Position: WR (Wide Receiver)
            - Zach Wilson – Position: QB (Quarterback)
            - Elijah Moore – Position: WR (Wide Receiver)
            - Michael Carter – Position: RB (Running Back)
            - Garrett Wilson – Position: WR (Wide Receiver)
            """
        )

    with defe:
        st.markdown(
            """
            On the other side, the Bills are on defense, and they're in a strong position. They're leading with a **77.84% win probability**, and their defense is using **man-to-man coverage**, meaning each defender is responsible for covering a specific offensive player. They’ve set up in a **4-3 defense formation**.
            """
        )

        st.markdown(
            """
            Here’s a look at the Bills' **defensive lineup**:
            - Von Miller – Position: OLB (Outside Linebacker)
            - Jordan Phillips – Position: DT (Defensive Tackle)
            - Tremaine Edmunds – Position: ILB (Inside Linebacker)
            - Siran Neal – Position: CB (Cornerback)
            - Ed Oliver – Position: DT (Defensive Tackle)
            - Jaquan Johnson – Position: FS (Free Safety)
            - Dane Jackson – Position: CB (Cornerback)
            - Gregory Rousseau – Position: SS (Strong Safety)
            - Damar Hamlin – Position: DF (Defensive Flex)
            - Terrel Bernard – Position: MLB (Middle Linebacker)
            - Christian Benford – Position: CB (Cornerback)
            """
        )

    st.markdown(
        """
        **I know doing all this math at once can be a bit tedious, but what if we could handle it with a single click, along with exploring multiple other :orange[scenario] & :violet[Player Tendency] report?** 
        
        ***Feel free to test and :orange[explore multiple] such :orange[scenarios] here at [https://perfectly-imperfect.streamlit.app/game_play_simulator](https://perfectly-imperfect.streamlit.app/game_play_simulator)***
        """
    )

st.markdown(
"""
**:violet[Model Prediction]** vs. **:orange[Actual Outcome]**: The scenario described above comes from **Week 9** of the 2022 NFL season, specifically ***gameId: 2022110606*** and ***playId: 1531***. Our model, at this point, was not trained on this specific situation. For reference, the actual play can be viewed in the game footage here: [Video Link](https://youtu.be/-mMjBSavIzk?si=IqdidYEO4xxCFJJG&t=248). In this game, the New York Jets (NYJ) gained **:orange[24 yards]** with a **:orange[short pass down the middle]**. Our model, when calculating its prediction, estimated that the offense would gain **:violet[21-30 yards]** in this scenario and predicted a **:violet[middle pass]** strategy as a *second choice*. While this is a positive result, it highlights areas for further improvement in the model's ability to predict offensive strategies with greater precision. 

Moreover, the best for this simulator is that we can add any team player and his resposnibly and can see the result and see the result for scouting decisssion
"""
)


with st.container(border=True):
    yds, star, desc = st.columns([4,4,3])

    with yds:
        st.image("assets/images/simulator_result_1.png",caption="Yards Gained Predictions")

    with star:
        st.image("assets/images/simulator_result_2.png",caption="Offensive Strategies Predictions")

    with desc:
        st.markdown(
            """
            The donut and bar charts show the predictions, but that's just the start. **Clicking the [button](https://perfectly-imperfect.streamlit.app/game_play_simulator)** provides deeper insights into the *game's situational context*, *key strengths of offense and defense*, *player performance*, *challenges facing*, and the *ratio* of players involved in *blocking, passing, running, and coverage*. 
            
            It also highlights offensive tendencies, historical trends in passing, and pass results across different field zones **midfield**, **own territory**, **own deep zone**, and **opponent red zone** all available in this **[:orange[simulator]](https://perfectly-imperfect.streamlit.app/game_play_simulator)**.
            """
        )


pre_snap, post_snap = st.columns(2)

with pre_snap:
    with st.container(border=True):
        st.image("assets/images/pre_snap.png",caption="Pre-Snap Situation")

with post_snap:
    with st.container(border=True):
        st.image("assets/images/post_snap.png",caption="Post-Snap Results")

st.markdown("**Note**: When interpreting the pre/post-snap visual, consider them as a mirrored version—if a player appears on the left side in the data, assume their position is on the right side on the field.")



with st.container(border=True):
    st.markdown("**2.. :violet[Dink] & :violet[Dunk] Guide**")
   
    col_7, col_8 = st.columns([4.5,5.5])
    with col_7:
        st.image("assets/images/dink_dunk.png",use_container_width=True, caption="Ridgeline Chart")

    with col_8:
         st.image("assets/images/dink_dunk_3.png",use_container_width=True)
        
    st.markdown(
        """
        After running the Dink & Dunk pipeline methodology, we cleaned the data for all 32 NFL teams and found a total of **8,704 pass plays**, with **82.81%** being **short passes**. Insights from ***Coach Lamberson*** guided me on why NFL teams are using these short passes also known as "**Dink and Dunk**". Based on this, we categorized them into types of short passes and found that *short middle, left, and right* were the most used also we extracted their *success rate, first down rate, average passing yards per play, and EPA*. The analysis showed that short middle passes had a ***Success Rate of 69.60%, a First Down Rate of 56.44%, Yards Per Play of 7.38 yards***, and an ***EPA Per Play of 0.18***.

        Going more granular, we analyzed individual teams and their players. The ridge line plot above illustrates the performance of the **Philadelphia Eagles**, showing that they were most efficient in ***short middle passes, with a median EPA of 0.90***.
        - **To interpret the plot**: the ***black line*** represents the ***mean line***, and the ***white line*** is the ***median***. When the mean is greater than the median in Expected Points Added (EPA), it suggests that there are a few high-value plays that significantly increase the average, while most plays are of lower value. Conversely, if the median is greater than the mean, it indicates that most plays are performing well, but there are some outliers (poor plays) that are dragging the average down.
        - The provided **table highlights** :violet[*receiver tendencies, showing total passes made to each receiver, their completion success rate, incomplete pass rate, preferred locations, most common routes, and their top five routes*]. This helps identify receiver tendencies and aids in scouting decisions. 
        
        Additionally, a **download button** is available to download the full report in PDF format for your desired team.

        :orange[**Scouting**] - :violet[**Dink**] **&** :violet[**Dunk**] **Dashboard** can be found [**https://perfectly-imperfect.streamlit.app/dink_dunk**](https://perfectly-imperfect.streamlit.app/dink_dunk)
        """
    )

st.divider()

st.markdown("#### Evaluation of Week 9 - :violet[Unseen Data]")
f1, f2, = st.columns(2)

with f1:
    with st.container(border=True):
        st.image("assets/images/nfl-big-data-bowl-2025/fold_1.png")

with f2:
    with st.container(border=True):
        st.image("assets/images/nfl-big-data-bowl-2025/fold_2.png")


f3, f4, f5 = st.columns(3)
with f3:
    with st.container(border=True):
        st.image("assets/images/nfl-big-data-bowl-2025/fold_3.png",caption="Fold - 3")

with f4:
    with st.container(border=True):
        st.image("assets/images/nfl-big-data-bowl-2025/fold_4.png",caption="Fold - 4")

with f5:
    with st.container(border=True):
        st.image("assets/images/nfl-big-data-bowl-2025/fold_5.png",caption="Fold - 5")

st.markdown(
    """
    Above 5-fold cross-validation result shows us that we have achieved above 80% accuracy & log-loss of 0.5 in predicting yards gained and strategy used. The average performance during training and validation further supports the model's robustness shown bellow

    **Training and Validation Performance**:
        → :violet[Average Training Loss]: ***0.5253***
        → :violet[Average Validation Loss]: ***0.5516***
        → :violet[Average Training Accuracy]: ***0.8288***
        → :violet[Average Validation Accuracy]: ***0.8265***

    """
)

result_1, result_2 =  st.columns(2)


with result_1:
    with st.container(border=True):
        st.markdown(
            """
            ##### :violet[Yards Gained] Prediction:
            - **Total Scenarios**: 1535
            - **Top 3 Matches** (Actual Yards in Top 3 Predictions): 1416 (**92.25%**)
            - **Top 2 Matches** (Actual Yards in Top 2 Predictions): 1277 (**83.19%**)
            - **Perfect Matches** (Actual Yards as Top Prediction): 969 (**63.13%**)
            - **Mismatches**: 119 (7.75%)
            """
        )


with result_2:
    with st.container(border=True):
        st.markdown(
            """
            ##### Strategy Prediction (:violet[Filtered Data]):
            - **Total Scenarios**: 807
            - **Top 3 Matches** (Actual Strategy in Top 3 Predictions): 636 (**78.81%**)
            - **Top 2 Matches** (Actual Strategy in Top 2 Predictions): 475 (**58.86%**)
            - **Perfect Matches** (Actual Strategy as Top Prediction): 245 (**30.36%**)
            - **Mismatches**: 171 (21.19%)
            """
        )

st.markdown(
    """
    These results validate strong predictive performance, with closely aligned training & validation metrics, in yards gained prediction, & highlight areas for improvement in strategy identification.
    """
)

st.divider()

st.markdown(
    """
    #### Conclusion
    Based on the two playgrounds discussed above:

    1. :violet[**Gameplay Scenario**]: We achieved 80% accuracy on Week 9's unseen data, which shows that we can rely on this initial model and the best part is that we can explore any player across 32 NFL teams or even within the same team to see if our game plan will work out or not. Moreover, video film is mandatory for scouting a player, but having this simulator is like the icing on the cake which makes things faster for us and helping us make decisions based on data.

    2. :violet[**Dink & Dunk Strategy**]: Regarding this strategy, the 2022 data indicates that more yards were gained in the middle location across all teams with higher pass completion rate, higher first down conversion rate and all this claims are backed up with the EPA median favoring short middle passes. So, consider using short middle passes to create pressure on defesne and gain more yards. However, it’s also important to analyze pass-receivers preferred locations and routes provided in the Dink and Dunk strategy report.

    Lastly, while I'd love to dive deeper, space constraints and the small sample size limit me. However, to sum it up, we have a comprehensive [**dashboard**](https://perfectly-imperfect.streamlit.app/) that breaks everything down in detail. It lets us explore custom players and game scenarios to determine if an offensive plan will work and understand which strategies might be used. With trends and patterns for both offense and defense, we can also filter data at the individual team level with ridge charts and tables that highlight tendencies and patterns, making data-driven decisions much easier.

    #### Acknowledgements
    Big thanks to the DFO (DeMarkco Butler) for all the help & support also I'm really thankful to Coach Lamberson for the live college football game insights they were super helpful for this project. 

    Thanks to Dr. Sam Ramanujan, Ryan Peterson, Shreya Raval, Kayla Baer, Shrey Raval, Rushabh Shah & Harsh Patel for their guidance & support.

    #### Appendix

    - Source Code can be found [**here**](https://github.com/Kush-Trivedi/NFL-Big-Data-Bowl-2025)
    - Kaggle Notebook can be found [**here**](https://www.kaggle.com/code/kushtrivedi14728/perfectly-imperfect)
    - A dashboard with comprehensive visuals and detailed player tendencies reports can be found [**here**](https://perfectly-imperfect.streamlit.app)
    - Professional LinkedIn [**profile**](https://www.linkedin.com/in/kush-trivedi/)

    Additionally, I understand that it might not be practical to navigate through all the links given the time constraints. While we were ***limited to submitting 20 slides, we still have 13 slides available***, so why not use them? These additional slides, though optional, include a **comprehensive scenario and a detailed 'Dink & Dunk' report**, presenting everything in a single, cohesive view. Feel free to review the following slides at your convenience. Thank you for taking the time to go through our work and for considering our submission.

    """
)
