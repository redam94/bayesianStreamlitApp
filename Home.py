import streamlit as st 

st.markdown("""
            # Bayesian Linear Regression
            ---
            ## 1. Upload and Preview Data
            Data should be cleaned and saved as a CSV file.\n
            The data should have a header row of variable names.\n
            Missing values should not be present.\n\n
            ## 2. Select Variables to use in Model
            Select the variables to use in the model.\n
            * <b>Media variables</b>: transformed using an S curve transformation <span style="color:red">TODO: Allow for other transformations and delays</span>
            * <b>Control variables</b>: untransformed variables <span style="color:red">TODO: Allow for transforms</span>
            * <b>Dependent variable</b>: the variable your trying to predict or fit the model to.
            * <b>Period variable</b>: Time variable not used during modeling.
            * <b>Geography variable</b>: Variable used to group data by geography. <span style="color:red">TODO: Allow for multiple grouping variables</span>
            ## 3. Model View
            View the model variable distributions.
            <span style="color:red">TODO: Show model predictions including media contributions</span>
            """, unsafe_allow_html=True)