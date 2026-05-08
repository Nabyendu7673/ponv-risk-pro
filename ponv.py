        # Apply the color function to the 'Score Contribution' column
        styled_df = df_scoring_breakdown.style.apply(lambda col: col.map(color_score), subset=['Score Contribution'])

        # Display the styled table
        st.table(styled_df)
