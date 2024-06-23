import os
import pandas as pd
import streamlit as st
import altair as alt
from matplotlib import pyplot as plt
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Filter,
    FilterExpression,
    Dimension,
    Metric,
    RunReportRequest,
)


st.set_page_config(
    page_title="Google Analytics Data Dashboard",
    page_icon="🚀",
    layout="wide",  
    initial_sidebar_state="auto", 
)

# Set up Streamlit app title
st.title("📈 SEO Data Dashboard")
st.divider()
st.sidebar.header("Add your filters here👇")

property_id = "346417629"

st.markdown("<h2><u>Google Analytics 4 Data</u><h2>", unsafe_allow_html=True)

# Date range input for the first date frame
start_date_1 = st.sidebar.date_input("Start date of current month", pd.to_datetime("2024-01-18"))
end_date_1 = st.sidebar.date_input("End date of current month", pd.to_datetime("today"))

# Date range input for the second date frame
start_date_2 = st.sidebar.date_input("Start date of month to compare", pd.to_datetime("2024-01-18"))
end_date_2 = st.sidebar.date_input("End date of month to compare", pd.to_datetime("today"))

# Run report request for the first date frame
client = BetaAnalyticsDataClient()
request_1 = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[Dimension(name="sessionDefaultChannelGroup")],
    metrics=[
        Metric(name="activeUsers"),
        Metric(name="newUsers"),
        Metric(name="engagedSessions"),
    ],
    date_ranges=[DateRange(start_date=start_date_1.strftime("%Y-%m-%d"), end_date=end_date_1.strftime("%Y-%m-%d"))],
    dimension_filter=FilterExpression(
        filter=Filter(
            field_name="sessionDefaultChannelGroup",
            string_filter={"value": "Organic Search"},
        )
    ),
)

response_1 = client.run_report(request_1)

# Run report request for the second date frame
request_2 = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[Dimension(name="sessionDefaultChannelGroup")],
    metrics=[
        Metric(name="activeUsers"),
        Metric(name="newUsers"),
        Metric(name="engagedSessions"),
    ],
    date_ranges=[DateRange(start_date=start_date_2.strftime("%Y-%m-%d"), end_date=end_date_2.strftime("%Y-%m-%d"))],
    dimension_filter=FilterExpression(
        filter=Filter(
            field_name="sessionDefaultChannelGroup",
            string_filter={"value": "Organic Search"},
        )
    ),
)

response_2 = client.run_report(request_2)

# Combine data into a single list
combined_data = []

for row in response_1.rows:
    combined_data.append({
        'Date Range': f"{start_date_1.strftime('%B %Y')}",
        'Channel': row.dimension_values[0].value,
        'Active Users': row.metric_values[0].value,
        'New Users': row.metric_values[1].value,
        'Engaged Sessions': row.metric_values[2].value
    })

for row in response_2.rows:
    combined_data.append({
        'Date Range': f"{start_date_2.strftime('%B %Y')}",
        'Channel': row.dimension_values[0].value,
        'Active Users': row.metric_values[0].value,
        'New Users': row.metric_values[1].value,
        'Engaged Sessions': row.metric_values[2].value
    })


# Create a single DataFrame
df_combined = pd.DataFrame(combined_data)

# ---
df = pd.read_csv('za.csv')

df['created'] = pd.to_datetime(df['created'])
#define how to aggregate various fields
agg_functions = {'created': 'first'}

df['count'] = df.groupby([df['created'].dt.year, df['created'].dt.month])['created'].transform('count')

df1 = df.groupby(df['created'].dt.month).size().reset_index(name='Conversions')

#df['created'].dt.month.value_counts()
# df['created'].dt.month

res = df.groupby(df['created'].dt.month).size()
#print(res)
res = df.groupby(df['created'].dt.month)['created'].count()
#print(res)
res = df.groupby(df['created'].dt.month).value_counts()
#print(res)
#df.groupby(df['created'].dt.month).agg({'count'})


# df = pd.DataFrame({'employee_id': [1, 1, 2, 3, 3, 3],
#                     'employee_name': ['Carlos', 'Carlos', 'Dan', 'Samuel', 'Samuel', 'Samuel'],
#                     'sales': [4, 1, 3, 2, 5, 3],})

#create new DataFrame by combining rows with same id values as_index = True
df_new = df.groupby([df['created'].dt.year, df['created'].dt.month]).aggregate("first")
df_new = df_new.rename(columns={'created': 'Date', 'count': 'Conversions'})
df_new.index.rename(['Year','Month'],inplace=True)
st.write(df_new)

st.write(df_new['Date'])

st.write(df_new['Conversions'])
# count sum of state in each month
#df.groupby(df.created.dt.month)['state'].sum()

#ct = df.groupby('created').size().values

#df
# df = df.drop_duplicates(subset="created").assign(Count=ct)

# cnt = df.groupby(pd.Grouper(key='created', axis=0, freq='M')).size().rename('Count')

# result = df.drop_duplicates(subset='created').merge(cnt, left_on='created', right_index=True)
# result
# @title Conversions

st.pyplot(df_new['Conversions'].plot(kind='line', figsize=(8, 5), title='Form submissions').figure)

#st.pyplot(plt.gca().spines[['top', 'right']].set_visible(False))

# ---

# st.title("Simulation[tm]")
# st.write("Here is our super important simulation")

# x = st.slider('Slope', min_value=0.01, max_value=0.10, step=0.01)
# y = st.slider('Noise', min_value=0.01, max_value=0.10, step=0.01)

# st.write(f"x={x} y={y}")
# values = np.cumprod(1 + np.random.normal(x, y, (100, 10)), axis=0)

# for i in range(values.shape[1]):
#     plt.plot(values[:, i])

# st.pyplot()

from matplotlib.dates import ConciseDateFormatter

fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
dates = np.arange(np.datetime64('2023-06-01'), np.datetime64('2024-06-24'),
                  np.timedelta64(1, 'h'))
# data = np.cumsum(np.random.randn(len(dates)))
st.write(df_new['Conversions'])
ax.plot([(2023,1), (2023,2), (2023,3), (2023,4)], [6,7,2,8])

#ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))

st.pyplot(fig)



# Display Combined DataFrame in Streamlit
st.subheader("Month on Month Data")
st.dataframe(df_combined)

# Bar Chart for Active Users, New Users, and Engaged Sessions
chart = alt.Chart(df_combined).mark_bar().encode(
    x=alt.X('Date Range:N', title='Date Range'),
    y=alt.Y('Active Users:Q', title='Count'),
    color=alt.Color('Channel:N', title='Channel'),
    column=alt.Column('Metric:N', title='Metrics')
).transform_fold(
    fold=['Active Users', 'New Users', 'Engaged Sessions'],
    as_=['Metric', 'Count']
)

# Display the chart in Streamlit
st.subheader("Active users MoM")
st.altair_chart(chart, use_container_width=True)

# Run report request for the top 10 landing pages
request_landing_pages = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[
        Dimension(name="landingPage"),
    ],
    metrics=[
        Metric(name="activeUsers"),
        Metric(name="newUsers"),
        Metric(name="engagedSessions"),
    ],
    date_ranges=[
        DateRange(start_date=start_date_1.strftime("%Y-%m-%d"), end_date=end_date_1.strftime("%Y-%m-%d"))
    ],
)

response_landing_pages = client.run_report(request_landing_pages)

# Extract top 10 landing pages
top_landing_pages_data = []
for row in response_landing_pages.rows[:10]:
    top_landing_pages_data.append({
        'Landing Page': row.dimension_values[0].value,
        'Active Users': row.metric_values[0].value,
        'New Users': row.metric_values[1].value,
        'Engaged Sessions': row.metric_values[2].value
    })

# Create DataFrame for top 10 landing pages
df_top_landing_pages = pd.DataFrame(top_landing_pages_data)

# Display DataFrame for top 10 landing pages in Streamlit
st.subheader("Top 10 Landing Pages")
st.dataframe(df_top_landing_pages)


st.markdown("<h2><u>Search Console Data</u><h2>", unsafe_allow_html=True)
#Run report request for the first GSC date frame
client = BetaAnalyticsDataClient()
request_1 = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[Dimension(name="yearMonth")],
    metrics=[
        Metric(name="organicGoogleSearchClicks"),
        Metric(name="organicGoogleSearchImpressions"),
        Metric(name="organicGoogleSearchClickThroughRate"),
    ],
    date_ranges=[DateRange(start_date=start_date_1.strftime("%Y-%m-%d"), end_date=end_date_1.strftime("%Y-%m-%d"))],
)

response_1 = client.run_report(request_1)

# Run report request for the second date frame
request_2 = RunReportRequest(
    property=f"properties/{property_id}",
    dimensions=[Dimension(name="yearMonth")],
    metrics=[
        Metric(name="organicGoogleSearchClicks"),
        Metric(name="organicGoogleSearchImpressions"),
        Metric(name="organicGoogleSearchClickThroughRate"),
    ],
    date_ranges=[DateRange(start_date=start_date_2.strftime("%Y-%m-%d"), end_date=end_date_2.strftime("%Y-%m-%d"))],
)

response_2 = client.run_report(request_2)

# Combine GSC data into a single list
combined_GSC_data = []

for row in response_1.rows:
    combined_GSC_data.append({
        'Month': row.dimension_values[0].value,
        'Clicks': row.metric_values[0].value,
        'Impressions': row.metric_values[1].value,
        'CTR': row.metric_values[2].value
    })

for row in response_2.rows:
    combined_GSC_data.append({
        'Month': row.dimension_values[0].value,
        'Clicks': row.metric_values[0].value,
        'Impressions': row.metric_values[1].value,
        'CTR': row.metric_values[2].value
    })

# Create a single DataFrame
df_GSC_combined = pd.DataFrame(combined_GSC_data)
st.subheader("Month on Month Data")
st.dataframe(df_GSC_combined)

#GSC Charts
st.subheader("Clicks Over Months")
gsc_chart = alt.Chart(df_GSC_combined).mark_bar().encode(
    x='Month:N',
    y='Clicks:Q',  
    color='Month:N'
).properties(
    width=600,
    height=400
)

st.altair_chart(gsc_chart, use_container_width=True)

st.subheader("Impressions Over Months")
gsc_chart = alt.Chart(df_GSC_combined).mark_bar().encode(
    x='Month:N',
    y='Impressions:Q',  
    color='Month:N'
).properties(
    width=600,
    height=400
)

st.altair_chart(gsc_chart, use_container_width=True)


