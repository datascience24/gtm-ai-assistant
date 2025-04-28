# --- Imports ---
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import openai
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import os

# --- Configurations ---
client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
st.set_page_config(page_title="GTM AI Sales Assistant", page_icon="🤖")
st.title("🤖 GTM AI Sales Assistant")

st.markdown("""
    <style>
        .block-container {
            padding-top: 5rem;
            padding-bottom: 4rem;
            padding-left: 2.5rem;
            padding-right: 2.5rem;
            background-color: #f9f9f9;
        }
        .stDownloadButton button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("🛠️ About this App")
    st.write(
        "This assistant helps sales teams prioritize accounts using a logistic regression model based on engagement, usage, and other factors. "
        "Upload your CSV to get started!"
    )
    
    st.header("💬 Example Questions You Can Ask")
    st.markdown("""
    - Why is Account Name not ready?
    - Why is Account Name so ready?
    - How can I improve Account Name's readiness?
    - What is the plan of action for Account Name?
    - What is the usage trend for Account Name?
    - What industry is Account Name in?
    """)

    st.header("📩 Contact")
    st.write("Questions? Reach out to the analytics team.")


# --- Session State Setup ---
if "accounts_df" not in st.session_state:
    st.session_state.accounts_df = None

# --- Functions ---
def classify_intent_with_openai(user_input):
    prompt = f"""
You are a strict JSON classification assistant. Given a user input from a sales team member, classify the intent into one of:
- get_readiness_scores
- explain_account_readiness
- suggest_account_improvement
- lookup_account_field
- other
Also extract the account name if mentioned.
⚡ Respond ONLY with valid compact JSON like:
{{"intent": "get_readiness_scores", "account": ""}}
{{"intent": "filter_accounts_by_field", "account": ""}}
{{"intent": "explain_account_readiness", "account": "BrightNetworks"}}
No commentary.
Now classify this input:
User Input: "{user_input}"
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result_text = response.choices[0].message.content.strip()
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        result = {"intent": "other", "account": ""}
    return result

# --- Main App ---
if uploaded_file := st.file_uploader("Upload your account CSV", type=["csv"]):
    if st.session_state.accounts_df is None:
        accounts_df = pd.read_csv(uploaded_file)

        le = LabelEncoder()
        accounts_df["Event Attendance Encoded"] = le.fit_transform(accounts_df["Event Attendance"])
        accounts_df["Usage Trend Encoded"] = le.fit_transform(accounts_df["Usage Trend"])
        accounts_df["Strategic Account Encoded"] = le.fit_transform(accounts_df["Strategic Account"])

        if "Expansion Potential" not in accounts_df.columns:
            accounts_df["Expansion Potential"] = (accounts_df["Rubrik ARR ($)"] / accounts_df["Revenue ($)"] < 0.1).astype(int)

        X = accounts_df[[
            "Engagement Score", "Company Size", "Revenue ($)", "Rubrik ARR ($)",
            "Event Attendance Encoded", "Usage Trend Encoded", "Strategic Account Encoded", "Expansion Potential"
        ]]

        accounts_df["Opportunity Created"] = np.random.randint(0, 2, size=len(accounts_df))
        y = accounts_df["Opportunity Created"]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression()
        model.fit(X_scaled, y)

        accounts_df["Readiness Score (%)"] = (model.predict_proba(X_scaled)[:, 1] * 100).round(1)

        # --- Generate Key Insights at Upload ---
        insights = []
        for idx, row in accounts_df.iterrows():
            account_info = row
            insight_prompt = f"""
Account Details for {account_info['Account Name']}:
- Industry: {account_info['Industry']}
- Revenue: {account_info['Revenue ($)']}
- Rubrik ARR: {account_info['Rubrik ARR ($)']}
- Engagement Score: {account_info['Engagement Score']}
- Usage Trend: {account_info['Usage Trend']}
- Event Attendance: {account_info['Event Attendance']}
- Renewal Date: {account_info['Renewal Date']}
- Strategic Account: {account_info['Strategic Account']}
- Expansion Potential: {account_info['Expansion Potential']}

Readiness Score: {account_info['Readiness Score (%)']}%

Write a very concise 1-sentence business summary for why this account is at this readiness score, highlighting 2-3 key factors and for the accounts with a higher score focus on the positives, for the accounts with a lower score focus on the negatives but include both positives and negatives in both cases.
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": insight_prompt}],
                    temperature=0.5
                )
                insight = response.choices[0].message.content.strip()
            except Exception as e:
                insight = "Insight generation error."

            insights.append(insight)

        accounts_df["Key Insight"] = insights

        st.session_state.accounts_df = accounts_df
    else:
        accounts_df = st.session_state.accounts_df

    # --- Executive Summary ---
    total_accounts = len(accounts_df)
    high_readiness = accounts_df[accounts_df["Readiness Score (%)"] >= 70]
    mid_readiness = accounts_df[(accounts_df["Readiness Score (%)"] >= 40) & (accounts_df["Readiness Score (%)"] < 70)]
    low_readiness = accounts_df[accounts_df["Readiness Score (%)"] < 40]

    high_revenue = high_readiness["Revenue ($)"].sum()
    mid_revenue = mid_readiness["Revenue ($)"].sum()
    low_revenue = low_readiness["Revenue ($)"].sum()

    st.markdown("## 📈 Executive Summary")
    st.markdown(f"""
    - **Total Accounts Scored:** {total_accounts}
    - **High Readiness Accounts (≥ 70%):** {len(high_readiness)} accounts totaling **${high_revenue:,.0f}** in revenue
    - **Medium Readiness Accounts (40%-70%):** {len(mid_readiness)} accounts totaling **${mid_revenue:,.0f}** in revenue
    - **Low Readiness Accounts (< 40%):** {len(low_readiness)} accounts totaling **${low_revenue:,.0f}** in revenue
    """)



    st.success("The accounts were evaluated with a logistic regression model. \n \n The key features are Usage Trend, Engagement Score, Industry, Revenue, ARR with Rubrik, Expansion Potential, and Renewal Date. \n \n  Here are the readiness scores and insights:")
    display_cols = [
        "Readiness Score (%)", "Account Name", "Key Insight", "Industry", "Revenue ($)", "Rubrik ARR ($)",
        "Engagement Score", "Event Attendance", "Renewal Date", "Usage Trend", "Strategic Account", "Expansion Potential"
    ]
    


    st.markdown("### 📊 Portfolio Overview")

    # Split into two columns
    col1, col2 = st.columns(2)

    # Left Chart: Number of Accounts per Readiness Bin
    with col1:
        accounts_df["Readiness Bin"] = pd.cut(accounts_df["Readiness Score (%)"], bins=[0,10,20,30,40,50,60,70,80,90,100])
        bin_counts = accounts_df["Readiness Bin"].value_counts().sort_index()

        fig, ax = plt.subplots()
        bin_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Readiness Score (%) Range')
        ax.set_ylabel('Number of Accounts')
        ax.set_title('📈 Accounts per Readiness Range')
        st.pyplot(fig)

    # Right Chart: Total Revenue per Readiness Bin
    with col2:
        bin_revenue = accounts_df.groupby("Readiness Bin")["Revenue ($)"].sum()

        fig2, ax2 = plt.subplots()
        bin_revenue.plot(kind='bar', ax=ax2)
        ax2.set_xlabel('Readiness Score (%) Range')
        ax2.set_ylabel('Total Revenue ($)')
        ax2.set_title('💵 Revenue per Readiness Range')
        st.pyplot(fig2)

# Then continue with your download buttons and tables...

    # --- 📥 Download Button ---
    st.download_button(
    label="📥 Download Prioritized Accounts (CSV)",
    data=accounts_df[display_cols].sort_values(by="Readiness Score (%)", ascending=False).to_csv(index=False),
    file_name="prioritized_accounts.csv",
    mime="text/csv"
)

    st.markdown("## 📋 Detailed Accounts Table")
    st.dataframe(accounts_df[display_cols].sort_values(by="Readiness Score (%)", ascending=False), use_container_width=True)

    st.markdown("---")
    st.header("Ask the AI Assistant about these accounts:")
    user_input = st.text_input("Ask me a question:")

    if user_input:
        classification = classify_intent_with_openai(user_input)
        intent = classification.get("intent", "other")
        account_name = classification.get("account", "")

        if intent == "get_readiness_scores":
            st.success("Here are the accounts ranked by readiness:")
            st.dataframe(accounts_df[display_cols].sort_values(by="Readiness Score (%)", ascending=False))
        

        elif intent == "explain_account_readiness" and account_name:
            account_row = accounts_df[accounts_df["Account Name"].str.lower() == account_name.lower()]
            if not account_row.empty:
                account_info = account_row.iloc[0]
                details = f"""
Account Details for {account_info['Account Name']}:
- Industry: {account_info['Industry']}
- Revenue: {account_info['Revenue ($)']}
- Rubrik ARR: {account_info['Rubrik ARR ($)']}
- Engagement Score: {account_info['Engagement Score']}
- Usage Trend: {account_info['Usage Trend']}
- Event Attendance: {account_info['Event Attendance']}
- Renewal Date: {account_info['Renewal Date']}
- Strategic Account: {account_info['Strategic Account']}
- Expansion Potential: {account_info['Expansion Potential']}

Readiness Score: {account_info['Readiness Score (%)']}%

Write a natural, concise explanation for a sales rep on why this account is at this readiness score, highlighting the most important factors.

Some explanation on the existing features in the dataset for context:
- Engagement Score is out of 100. A score below 50 is considered weak and should be improved through customer success engagement. Closer to 0 is really bad and closer to 100 is really good.
- If Event Attendance is No, recommend inviting the account to relevant events to boost engagement.
- If Usage Trend is Down, suggest promoting product feature usage and providing enablement training.
- If Expansion Potential = 0, it indicates the account already spends a large proportion (>10%) of its revenue on Rubrik, suggesting strong current commitment. Upsell opportunities might be limited unless new product offerings are introduced.
- If Expansion Potential = 1, there is opportunity to upsell based on unused budget.
- If Renewal Date is within the next 90 days, that is a positive opportunity for renewal discussions.

"""
                try:
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": details}],
                        temperature=0.5
                    )
                    dynamic_insight = response.choices[0].message.content.strip()
                    st.success(dynamic_insight)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")
            else:
                st.error("Account not found.")

        elif intent == "suggest_account_improvement" and account_name:
            account_row = accounts_df[accounts_df["Account Name"].str.lower() == account_name.lower()]
            if not account_row.empty:
                account_info = account_row.iloc[0]
                improvement_prompt = f"""
The following is the current status of the account {account_info['Account Name']}:
- Industry: {account_info['Industry']}
- Revenue: {account_info['Revenue ($)']}
- Rubrik ARR: {account_info['Rubrik ARR ($)']}
- Engagement Score: {account_info['Engagement Score']}
- Usage Trend: {account_info['Usage Trend']}
- Event Attendance: {account_info['Event Attendance']}
- Renewal Date: {account_info['Renewal Date']}
- Strategic Account: {account_info['Strategic Account']}
- Expansion Potential: {account_info['Expansion Potential']}

The readiness score is {account_info['Readiness Score (%)']}%.

Guidelines for improvement:
- Engagement Score is out of 100. A score below 50 is considered weak and should be improved through customer success engagement. Closer to 0 is really bad and closer to 100 is really good.
- If Event Attendance is No, recommend inviting the account to relevant events to boost engagement.
- If Usage Trend is Down, suggest promoting product feature usage and providing enablement training.
- If Expansion Potential = 0, it indicates the account already spends a large proportion (>10%) of its revenue on Rubrik, suggesting strong current commitment. Upsell opportunities might be limited unless new product offerings are introduced.
- If Expansion Potential = 1, there is opportunity to upsell based on unused budget.
- If Renewal Date is within the next 90 days, that is a positive opportunity for renewal discussions.

Based on these guidelines and the account's current data, write 2-3 concise sentences suggesting how the account's readiness could be improved.
"""
                try:
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": improvement_prompt}],
                        temperature=0.5
                    )
                    improvement_advice = response.choices[0].message.content.strip()
                    st.success(improvement_advice)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")
            else:
                st.error("Account not found.")

        elif intent == "lookup_account_field" and account_name:
            account_row = accounts_df[accounts_df["Account Name"].str.lower() == account_name.lower()]
            if not account_row.empty:
                target_field = None
                lowered = user_input.lower()
                if "usage trend" in lowered:
                    target_field = "Usage Trend"
                elif "industry" in lowered:
                    target_field = "Industry"
                elif "revenue" in lowered and "rubrik" not in lowered:
                    target_field = "Revenue ($)"
                elif "rubrik" in lowered:
                    target_field = "Rubrik ARR ($)"
                elif "event attendance" in lowered:
                    target_field = "Event Attendance"
                elif "renewal" in lowered:
                    target_field = "Renewal Date"

                if target_field:
                    value = account_row[target_field].values[0]
                    st.success(f"{account_name}'s {target_field} is {value}.")
                else:
                    st.error("I'm not sure which attribute you're asking about.")
            else:
                st.error("Account not found.")
                
        elif intent == "filter_accounts_by_field":
            lowered = user_input.lower()
            matched = False

            # Mapping of possible fields
            field_mappings = {
                "usage trend": "Usage Trend",
                "strategic account": "Strategic Account",
                "event attendance": "Event Attendance",
                "expansion potential": "Expansion Potential",
                "renewal date": "Renewal Date",
                "industry": "Industry"
            }

            for keyword, field_name in field_mappings.items():
                if keyword in lowered:
                    if "positive" in lowered or "yes" in lowered or "good" in lowered:
                        filtered = accounts_df[accounts_df[field_name].str.lower().isin(["positive", "yes", "good"])]
                    elif "down" in lowered or "negative" in lowered or "no" in lowered:
                        filtered = accounts_df[accounts_df[field_name].str.lower().isin(["down", "negative", "no"])]
                    else:
                        # Just show all accounts for that field without value filtering
                        filtered = accounts_df[~accounts_df[field_name].isnull()]

                    st.success(f"Filtered Accounts based on '{field_name}': {len(filtered)} accounts")
                    st.dataframe(filtered[display_cols])
                    matched = True
                    break

            if not matched:
                st.info("I'm not sure which field you want to filter by. Try mentioning 'usage trend', 'strategic account', 'event attendance', etc.")


        else:
            st.info("I'm currently focused on analyzing readiness and suggesting sales actions. Please ask about account insights!")



else:
    st.info("👆 Please upload your account data CSV to get started.")