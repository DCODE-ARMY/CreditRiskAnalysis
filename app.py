# Import libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import io
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from MarketCreditRiskAnalyzerCrew import MarketCreditRiskAnalyzerCrew, MarketDataOutput, GlobalEventsOutput, PeopleReviewsOutput, CreditRiskAssessmentReport, StrategicDecisionOutput
from io import StringIO
import sys
import time
import asyncio
from crewai import Crew
import re



# Initialize crew
# Initialize crew and create instance
crew = MarketCreditRiskAnalyzerCrew()
crew_instance = crew.crew()

# Custom class to redirect stdout
class StreamlitLogCapture:
    def __init__(self):
        self.buffer = StringIO()
        self.last_position = 0
        self.last_content = ""

    def write(self, text):
        self.buffer.write(text)
        self.buffer.flush()

    def flush(self):
        pass

    def get_new_logs(self):
        """Get only the new log content since last read"""
        current_pos = self.buffer.tell()
        self.buffer.seek(self.last_position)
        new_content = self.buffer.read()
        self.last_position = current_pos
        self.buffer.seek(current_pos)
        return new_content

    def get_all_logs(self):
        """Get all logs from the beginning"""
        current_pos = self.buffer.tell()
        self.buffer.seek(0)
        content = self.buffer.getvalue()
        self.buffer.seek(current_pos)
        return content




async def async_crew_execution(company, log_capture, log_placeholder, status_placeholder):
    """Execute crew asynchronously and handle real-time updates"""
    try:
        # Initialize status
        status_placeholder.info(f"🚀 Starting analysis for {company}")

        # Start the crew operation asynchronously
        result = await crew_instance.kickoff_async({'company': str(company)})

        # Process the results
        market_data = result.tasks_output[0].pydantic
        events = result.tasks_output[1].pydantic
        reviews = result.tasks_output[2].pydantic
        credit_risk = result.tasks_output[3].pydantic
        strategic_decision = result.tasks_output[4].pydantic

        # Final updates
        all_logs = log_capture.get_all_logs()
        log_placeholder.text_area("Agent Logs (Final)", all_logs, height=300)
        st.session_state['logs'] = all_logs
        status_placeholder.success("✅ Analysis completed successfully!")

        return type('CrewResult', (), {'tasks_output': [
            type('TaskOutput', (), {'output': market_data}),
            type('TaskOutput', (), {'output': events}),
            type('TaskOutput', (), {'output': reviews}),
            type('TaskOutput', (), {'output': credit_risk}),
            type('TaskOutput', (), {'output': strategic_decision})
        ]})
    except Exception as e:
        st.error(f"Failed to process data for {company}: {str(e)}")

        all_logs = log_capture.get_all_logs()
        if all_logs:
            log_placeholder.text_area("Agent Logs (Error State)", all_logs, height=300)
            st.session_state['logs'] = all_logs

        status_placeholder.error("❌ Analysis failed. Please try again.")
        return None

def extract_status_from_logs(new_log_content):
    """Extract the most recent task-agent-tool mapping from logs"""
    if not new_log_content:
        return None

    status_updates = []
    # Known agents and tasks for specific matching
    agents = [
        "credit_researcher_agent",
        "credit researcher agent",
        "credit_risk_agent",
        "credit risk agent",
        "strategic_decision_agent",
        "strategic decision agent",
        "decision_support_agent"
        "decision support agent",
        "Crew Manager",
        "crew manager"
    ]
    tasks = [
        "market_data_task",
        "market data task",
        "global_events_task",
        "global events task",
        "people_reviews_task",
        "people reviews task",
        "credit_risk_task",
        "credit risk task",
        "strategic_decision_task"
        "strategic decision task",
    ]

    # Keep track of the last agent name for tool or status context
    last_agent_name = "Agent"

    for line in new_log_content.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Identify tasks by ID or specific task name
        task_match = re.search(r"📋 Task: ([a-f0-9-]+)", line)
        if task_match:
            task_id = task_match.group(1)
            # Attempt to map task ID to a known task name (if logs provide context)
            task_name = next((t for t in tasks if t in line.lower()), "Task")
            status_updates.append(f"⚡ Task {task_name} (ID: {task_id}) is now executing...")
        else:
            # Check for specific task names directly in the line
            for task in tasks:
                if task.lower() in line.lower():
                    status_updates.append(f"⚡ Task {task.replace('_task', '').replace('_', ' ').title()} is now executing...")
                    break

        # Identify agents
        agent_match = re.search(r"🤖 Agent: (\w+)", line)
        if agent_match:
            agent_name_raw = agent_match.group(1)
            # Match to known agents, preserving case for display
            agent_name = next(
                (a.replace("_agent", "").replace("_", " ").title() for a in agents if a.lower() == agent_name_raw.lower()),
                agent_name_raw.replace("_", " ").title()
            )
            last_agent_name = agent_name
            status_updates.append(f"🤖 {agent_name} is currently active")
        else:
            # Check for specific agent names directly in the line
            for agent in agents:
                if agent.lower() in line.lower():
                    agent_name = agent.replace("_agent", "").replace("_", " ").title()
                    last_agent_name = agent_name
                    status_updates.append(f"🤖 {agent_name} is currently active")
                    break

        # Identify tool usage
        tool_match = re.search(r"Using (\w+_tool)", line)
        if tool_match:
            tool_name = tool_match.group(1).replace("_", " ").title()
            status_updates.append(f"🛠️ {last_agent_name} is using {tool_name}")

        # Additional status indicators
        if "Thinking" in line:
            status_updates.append(f"💡 {last_agent_name} is thinking...")
        elif "In Progress" in line:
            status_updates.append(f"🚀 {last_agent_name} is executing a task...")

    return status_updates[-1] if status_updates else None

def run_crew(company):
    log_capture = StreamlitLogCapture()
    original_stdout = sys.stdout
    sys.stdout = log_capture

    log_placeholder = st.empty()  # Placeholder for logs
    status_placeholder = st.empty()  # Placeholder for statuses
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def update_logs():
            """Continuously fetch logs and update the placeholder"""
            while True:
                new_logs = log_capture.get_new_logs()
                if new_logs:
                    all_logs = log_capture.get_all_logs()
                    log_placeholder.text_area("Agent Logs (Live)", all_logs, height=300)
                    st.session_state['logs'] = all_logs

                    # Extract and display real-time status updates
                    status = extract_status_from_logs(new_logs)
                    if status:
                        status_placeholder.info(status)

                await asyncio.sleep(0.5)  # Refresh every 500ms

        # Run log update in parallel
        log_task = loop.create_task(update_logs())

        # Start the async execution
        result = loop.run_until_complete(
            async_crew_execution(company, log_capture, log_placeholder, status_placeholder)
        )

        log_task.cancel()  # Stop updating logs after completion

        return result
    finally:
        sys.stdout = original_stdout
        loop.close()



# --------------------
# PDF Generation Function
# --------------------
def generate_pdf_report(result, company):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"Credit Risk Assessment Report for {company}", styles['Title']))
    story.append(Spacer(1, 12))

    # Overview
    story.append(Paragraph("Overview", styles['Heading2']))
    story.append(Paragraph(result.tasks_output[3].output.overall_summary or "No summary available", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Market Data
    story.append(Paragraph("Market Data", styles['Heading2']))
    market_data = result.tasks_output[0].output
    story.append(Paragraph(market_data.overview, styles['BodyText']))
    data = [["Metric", "Latest Value", "Historical Trend"]] + [
        [m.name, str(m.latest_value) if m.latest_value is not None else "N/A", m.historical_trend or "Not provided"]
        for m in market_data.metrics
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Paragraph(f"Historical vs Current: {market_data.historical_vs_current}", styles['BodyText']))
    story.append(Paragraph(f"Company Filings Analysis: {market_data.company_filings_analysis_narrative}", styles['BodyText']))
    story.append(Paragraph(f"Sources: {', '.join(market_data.sources)}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Events
    story.append(Paragraph("Events", styles['Heading2']))
    events_data = result.tasks_output[1].output
    story.append(Paragraph(events_data.overview, styles['BodyText']))
    for event in events_data.events:
        story.append(Paragraph(f"{event.date}: {event.event_title} ({event.event_type})", styles['Heading3']))
        story.append(Paragraph(event.impact_analysis, styles['BodyText']))
        story.append(Paragraph(f"Source: {event.source}", styles['BodyText']))
        story.append(Spacer(1, 6))
    story.append(Paragraph(f"Comprehensive Analysis: {events_data.comprehensive_analysis}", styles['BodyText']))
    story.append(Paragraph(f"Summary: {events_data.summary_of_findings}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # People Reviews
    story.append(Paragraph("People Reviews", styles['Heading2']))
    reviews_data = result.tasks_output[2].output
    story.append(Paragraph(reviews_data.overview, styles['BodyText']))
    story.append(Paragraph(f"Positive: {reviews_data.positive_sentiment.theme}", styles['Heading3']))
    for quote in reviews_data.positive_sentiment.evidence:
        story.append(Paragraph(f"- {quote}", styles['BodyText']))
    story.append(Paragraph(f"Negative: {reviews_data.negative_sentiment.theme}", styles['Heading3']))
    for quote in reviews_data.negative_sentiment.evidence:
        story.append(Paragraph(f"- {quote}", styles['BodyText']))
    story.append(Paragraph(f"Detailed Analysis: {reviews_data.detailed_analysis}", styles['BodyText']))
    story.append(Paragraph(f"Overall Assessment: {reviews_data.overall_assessment}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Credit Risk Report
    story.append(Paragraph("Credit Risk Report", styles['Heading2']))
    credit_risk = result.tasks_output[3].output
    story.append(Paragraph(f"Final Score: {credit_risk.final_credit_risk_score}/10", styles['BodyText']))
    story.append(Paragraph(f"Explanation of Score Calculation: {credit_risk.explanation_of_score_calculation}", styles['BodyText']))
    story.append(Paragraph(f"Detailed Explanation: {credit_risk.detailed_explanation}", styles['BodyText']))
    data = [["Component", "Value", "Risk Contribution"]] + [
        [entry.component, str(entry.value) if entry.value is not None else "N/A", str(entry.risk_contribution) if entry.risk_contribution is not None else "N/A"]
        for entry in credit_risk.details_table
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer

# --------------------
# Display Functions
# --------------------
def display_overview(result, key_suffix=""):
    st.header("Overview")
    credit_risk = result.tasks_output[3].output.final_credit_risk_score
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=credit_risk,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Credit Risk Score"},
            gauge={
                'axis': {'range': [1, 10]},
                'bar': {'color': "#2E86C1"},
                'steps': [
                    {'range': [1, 4], 'color': "#3498DB"},
                    {'range': [4, 7], 'color': "#D6EAF8"},
                    {'range': [7, 10], 'color': "#1B263B"}
                ]
            }
        )
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)")
    st.plotly_chart(fig, use_container_width=True, key=f"overview_chart_{key_suffix}")
    st.write(result.tasks_output[3].output.overall_summary or "No summary provided.")

def display_market_data(market_data, key_suffix=""):
    st.header("Market Data")
    st.write(market_data.overview)
    
    # Display metrics in a table
    df = pd.DataFrame(
        [
            (m.name, m.latest_value if m.latest_value is not None else "N/A", m.historical_trend or "Not provided")
            for m in market_data.metrics
        ],
        columns=["Metric", "Latest Value", "Historical Trend"]
    )
    st.markdown(
        f"<div style='overflow-x: auto; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>"
        f"<table style='width: 100%; border-collapse: collapse; background-color: #D6EAF8;'>"
        f"<thead><tr style='background-color: #2E86C1; color: #FFFFFF;'>"
        f"<th style='padding: 12px;'>{df.columns[0]}</th>"
        f"<th style='padding: 12px;'>{df.columns[1]}</th>"
        f"<th style='padding: 12px;'>{df.columns[2]}</th>"
        f"</tr></thead><tbody>"
        + ''.join([
            f"<tr style='border-bottom: 1px solid #3498DB;'>"
            f"<td style='padding: 12px;'>{row[0]}</td>"
            f"<td style='padding: 12px;'>{row[1]}</td>"
            f"<td style='padding: 12px;'>{row[2]}</td>"
            f"</tr>"
            for row in df.itertuples(index=False)
        ])
        + "</tbody></table></div>",
        unsafe_allow_html=True
    )

    # Prepare data for radar chart, filter out non-numeric or N/A values
    numeric_df = df[df["Latest Value"].apply(lambda x: isinstance(x, (int, float)) and x != "N/A")].copy()
    if not numeric_df.empty:
        # Default normalization range for unknown metrics
        default_range = (0, 100)  # Fallback for metrics without predefined ranges
        
        # Extended normalization ranges for known metrics (updated dynamically)
        normalization_ranges = {
           
            "Free Cash Flow": (-60000000000, 60000000000),  # Wide range for cash flows
            "Operating Cash Flow": (-50000000000, 50000000000),
            "Market Liquidity Measure": (0, 100000000),  # Shares or volume
            "MarketCapitalization": (0, 1000000000000),  # Updated for larger companies like Tesla
            "Altman Z-Score": (0, 15),             # Extended for high scores (e.g., Tesla’s 12.68)
            "Piotroski Score": (0, 9),
            "Return on Equity": (0, 100),          # Percentage
            "Return on Assets": (0, 100),          # Percentage
            "Price to Earnings Score": (0, 5),     # Score-based
            "Price to Book Score": (0, 5),         # Score-based
            "Enterprise Value TTM": (0, 1000000000000),  # Large range for EV
            "Current Ratio TTM": (0, 5),           # Ratio
            "Working Capital TTM": (-50000000000, 50000000000),  # Wide range
        }

        # Normalize values dynamically
        numeric_df["Normalized Value"] = numeric_df.apply(
            lambda row: (
                (row["Latest Value"] - normalization_ranges.get(row["Metric"], default_range)[0]) /
                (normalization_ranges.get(row["Metric"], default_range)[1] - normalization_ranges.get(row["Metric"], default_range)[0]) * 100
                if normalization_ranges.get(row["Metric"], default_range)[1] > normalization_ranges.get(row["Metric"], default_range)[0]
                else 0
            ),
            axis=1
        )

        # Create radar chart
        categories = numeric_df["Metric"].tolist()
        values = numeric_df["Normalized Value"].tolist()
        values += values[:1]  # Close the radar chart
        categories += categories[:1]

        fig = go.Figure(
            data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"{st.session_state.get('company', 'Company')} Financial Metrics",
                line_color="#2E86C1"
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(color="#1B263B")
                ),
                angularaxis=dict(
                    tickfont=dict(color="#1B263B")
                )
            ),
            paper_bgcolor="rgba(255,255,255,0.95)",
            plot_bgcolor="rgba(255,255,255,0.95)",
            title="Radar Chart of Normalized Key Financial Metrics",
            title_font_color="#1B263B",
            title_font_size=16,
            showlegend=True
        )
        # Add annotation with raw values
        raw_values = [f"{m}={v if isinstance(v, (int, float)) else 'N/A'}" for m, v in zip(df["Metric"], df["Latest Value"])]
        fig.add_annotation(
            text="Raw Values: " + ", ".join(raw_values),
            xref="paper", yref="paper",
            x=0.5, y=-0.2,
            showarrow=False,
            font=dict(size=12, color="#1B263B")
        )
        st.plotly_chart(fig, use_container_width=True, key=f"market_data_chart_{key_suffix}")
    
    with st.expander("Historical vs Current Analysis"):
        st.write(market_data.historical_vs_current)
    with st.expander("Filling Analysis"):
        st.write(market_data.company_filings_analysis_narrative)
    st.write("**Sources**: " + ", ".join(market_data.sources))

# ... (previous imports and functions remain unchanged)

def display_events(events_data, key_suffix=""):
    st.header("Events Analysis")
    st.write(events_data.overview)
    
    event_filter = st.selectbox("Filter by Event Type", ["All", "Global", "Local", "Internal"], key=f"event_filter_{key_suffix}")
    filtered_events = events_data.events if event_filter == "All" else [
        e for e in events_data.events if e.event_type == event_filter
    ]
    
    if not filtered_events:
        st.write("No events available for the selected filter.")
        return

    df = pd.DataFrame(
        [(e.date, e.event_title, e.event_type, e.impact_analysis, e.source) for e in filtered_events],
        columns=["Date", "Title", "Type", "Impact", "Source"]
    )

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['End_Date'] = df['Date'] + pd.Timedelta(days=2)  # Increased duration to 2 days

    if df['Date'].isna().all():
        st.write("Error: All dates are invalid or missing. Please check the input data.")
        return

    fig = px.timeline(
        df,
        x_start="Date",
        x_end="End_Date",
        y="Title",
        color="Type",
        color_discrete_map={"Global": "#2E86C1", "Local": "#3498DB", "Internal": "#D6EAF8"},
        title="Event Timeline"
    )
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0.95)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        font_color="#1B263B",
        title_font_color="#1B263B",
        title_font_size=16,
        xaxis_title="Date",
        yaxis_title="Event Title",
        xaxis=dict(
            rangeslider=dict(visible=True),
            range=[pd.to_datetime('2025-01-01'), pd.to_datetime('2025-03-01')]
        ),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis=dict(automargin=True)
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(width=0.2)
    st.plotly_chart(fig, use_container_width=True, key=f"events_chart_{key_suffix}")
    with st.expander("Comprehensive Analysis"):
        st.write(events_data.comprehensive_analysis)
    with st.expander("Summary of Findings"):
        st.write(events_data.summary_of_findings)

def display_reviews(reviews_data):
    st.header("People Reviews")
    st.write(reviews_data.overview)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive Sentiment", help="Reflects employee and public praise")
        st.markdown(
            f"<div style='background-color: #3498DB; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); color: #FFFFFF;'>"
            f"<strong>Theme:</strong> {reviews_data.positive_sentiment.theme}</div>",
            unsafe_allow_html=True
        )
        for quote in reviews_data.positive_sentiment.evidence:
            st.markdown(f"<p style='color: #3498DB; margin: 5px 0;'>{quote}</p>", unsafe_allow_html=True)
    with col2:
        st.subheader("Negative Sentiment", help="Highlights customer and operational concerns")
        st.markdown(
            f"<div style='background-color: #2E86C1; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); color: #FFFFFF;'>"
            f"<strong>Theme:</strong> {reviews_data.negative_sentiment.theme}</div>",
            unsafe_allow_html=True
        )
        for quote in reviews_data.negative_sentiment.evidence:
            st.markdown(f"<p style='color: #2E86C1; margin: 5px 0;'>{quote}</p>", unsafe_allow_html=True)
    with st.expander("Detailed Analysis"):
        st.write(reviews_data.detailed_analysis)
    with st.expander("Overall Assessment"):
        st.write(reviews_data.overall_assessment)

def display_credit_risk(risk_data, key_suffix=""):
    st.header("Credit Risk Report")
    st.write(f"**Final Credit Risk Score**: {risk_data.final_credit_risk_score}/10")
    st.write("**Explanation of Score Calculation**:")
    st.write(risk_data.explanation_of_score_calculation)
    df = pd.DataFrame(
        [
            (entry.component, entry.value if entry.value is not None else "N/A", entry.risk_contribution if entry.risk_contribution is not None else "N/A")
            for entry in risk_data.details_table
        ],
        columns=["Component", "Value", "Risk Contribution"]
    )
    st.markdown(
        f"<div style='overflow-x: auto; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>"
        f"<table style='width: 100%; border-collapse: collapse; background-color: #FFFFFF;'>"
        f"<thead><tr style='background-color: #2E86C1; color: #FFFFFF;'>"
        f"<th style='padding: 12px;'>{df.columns[0]}</th>"
        f"<th style='padding: 12px;'>{df.columns[1]}</th>"
        f"<th style='padding: 12px;'>{df.columns[2]}</th>"
        f"</tr></thead><tbody>"
        + ''.join([
            f"<tr style='border-bottom: 1px solid #3498DB;'>"
            f"<td style='padding: 12px;'>{row[0]}</td>"
            f"<td style='padding: 12px;'>{row[1]}</td>"
            f"<td style='padding: 12px;'>{row[2]}</td>"
            f"</tr>"
            for row in df.itertuples(index=False)
        ])
        + "</tbody></table></div>",
        unsafe_allow_html=True
    )
    fig = px.bar(
        df,
        x="Component",
        y="Risk Contribution",
        title="Risk Contribution by Component",
        color="Risk Contribution",
        color_continuous_scale=["#2E86C1", "#3498DB", "#D6EAF8"]
    )
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0.95)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        font_color="#1B263B",
        title_font_color="#1B263B",
        title_font_size=16
    )
    st.plotly_chart(fig, use_container_width=True, key=f"credit_risk_chart_{key_suffix}")
    with st.expander("Detailed Explanation"):
        st.write(risk_data.detailed_explanation)

def display_strategic_decisions(strategic_data, key_suffix=""):
    st.header("Strategic Decisions")
    st.write(strategic_data.overview)
    
    # Display decisions in a table
    df = pd.DataFrame(
        [
            (
                d.type.capitalize(),
                d.content,
                d.rationale,
                f"£{d.value:,.2f}" if d.value is not None else "N/A",
                d.deadline.strftime("%Y-%m-%d") if d.deadline else "N/A",
                ", ".join(d.dependencies) if d.dependencies else "None"
            )
            for d in strategic_data.decisions
        ],
        columns=["Type", "Content", "Rationale", "Value", "Deadline", "Dependencies"]
    )
    st.markdown(
        f"<div style='overflow-x: auto; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);'>"
        f"<table style='width: 100%; border-collapse: collapse; background-color: #D6EAF8;'>"
        f"<thead><tr style='background-color: #2E86C1; color: #FFFFFF;'>"
        f"<th style='padding: 12px;'>Type</th>"
        f"<th style='padding: 12px;'>Content</th>"
        f"<th style='padding: 12px;'>Rationale</th>"
        f"<th style='padding: 12px;'>Value</th>"
        f"<th style='padding: 12px;'>Deadline</th>"
        f"</tr></thead><tbody>"
        + ''.join([
            f"<tr style='border-bottom: 1px solid #3498DB;'>"
            f"<td style='padding: 12px;'>{row[0]}</td>"
            f"<td style='padding: 12px;'>{row[1]}</td>"
            f"<td style='padding: 12px;'>{row[2]}</td>"
            f"<td style='padding: 12px;'>{row[3]}</td>"
            f"<td style='padding: 12px;'>{row[4]}</td>"
            f"</tr>"
            for row in df.itertuples(index=False)
        ])
        + "</tbody></table></div>",
        unsafe_allow_html=True
    )

    # Bar chart for decision types with values
    value_df = df[df["Value"] != "N/A"].copy()
    if not value_df.empty:
        value_df["Value"] = value_df["Value"].str.replace("£", "").str.replace(",", "").astype(float)
        fig = px.bar(
            value_df,
            x="Type",
            y="Value",
            title="Strategic Decisions by Value",
            color="Type",
            color_discrete_sequence=["#2E86C1", "#3498DB", "#D6EAF8"],
            text=value_df["Value"].apply(lambda x: f"£{x:,.0f}")
        )
        fig.update_traces(textposition='auto')
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0.95)",
            plot_bgcolor="rgba(255,255,255,0.95)",
            font_color="#1B263B",
            title_font_color="#1B263B",
            title_font_size=16,
            yaxis_title="Value (£)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key=f"strategic_decisions_chart_{key_suffix}")

    # Expanders for additional details
    with st.expander("Tool Log"):
        st.write("\n".join(strategic_data.tool_log) if strategic_data.tool_log else "No tools used.")
    with st.expander("Assumptions"):
        st.write("\n".join(strategic_data.assumptions) if strategic_data.assumptions else "No assumptions made.")
    with st.expander("Next Steps"):
        st.write(strategic_data.next_steps if strategic_data.next_steps else "No next steps provided.")




# --------------------
# Main UI
# --------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="centered")
st.markdown(
    """
        <style>
          /* Color Palette:
            1) White (#FFFFFF)
            2) Light Blue (#D6EAF8)
            3) Medium Blue (#3498DB)
            4) Dark Blue (#2E86C1)
            5) Deep Blue (#1B263B)
          */
        
          body, .stApp {
              background-color: #FFFFFF;
              color: #1B263B;
              font-family: "Helvetica Neue", Arial, sans-serif;
              margin: 0;
              padding: 0;
          }
        
          .header {
              text-align: center;
              padding: 20px;
              border-bottom: 2px solid #D6EAF8;
          }
          .header h1 {
              margin: 0;
              font-size: 2.8em;
              color: #1B263B;
              text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
          }
          .logo {
              max-width: 150px;
              margin-bottom: 10px;
          }
        
          .stTabs [data-baseweb="tab-list"] {
              background-color: #D6EAF8;
              border: none;
              margin-top: 20px;
              padding: 3px;
              display: flex;           
              overflow-x: auto;
              border-radius: 8px;
              box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
              overflow-x: hidden;
              white-space: nowrap;
          }
          .stTabs [data-baseweb="tab"] {
              color: #1B263B;
              padding: 10px 9px;
              flex: 1;
              border: none;
              background: transparent;
              transition: background-color 0.3s, color 0.3s;
              display: inline-block;
              font-size: 16px;
              font-weight: 500;
          }
          .stTabs [data-baseweb="tab"]:hover {
              background-color: #3498DB;
              color: #FFFFFF;
              border-radius: 5px;
          }
          .stTabs [data-baseweb="tab"][aria-selected="true"] {
              background-color: #2E86C1;
              color: #FFFFFF;
              border-radius: 5px;
          }
          .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
              display: none;
          }
          .stTabs [data-baseweb="tab-list"] {
              -ms-overflow-style: none;
              scrollbar-width: none;
          }
        
          .main-content {
              width: 100%;
              max-width: 900px;
              margin: 30px auto;
              padding: 20px;
              background-color: #FFFFFF;
              border: 1px solid #D6EAF8;
              border-radius: 10px;
              box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          }
          .main-content h2, .main-content h3 {
              color: #1B263B;
              font-weight: 600;
          }
          .main-content p {
              line-height: 1.6;
              margin-bottom: 1em;
              color: #1B263B;
          }
        
          .accent-medium {
              background-color: #3498DB;
              color: #FFFFFF;
              padding: 15px;
              border-radius: 8px;
              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          }
          .accent-dark {
              background-color: #2E86C1;
              color: #FFFFFF;
              padding: 15px;
              border-radius: 8px;
              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          }
        
          .input-container {
              background-color: #D6EAF8;
              padding: 25px;
              border-radius: 15px;
              box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
              width: 100%;
              max-width: 700px;
              margin: 0 auto 30px auto;
          }
          .stTextInput div div input {
              border-radius: 10px;
              border: 2px solid #2E86C1;
              padding: 12px 40px 12px 15px;
              background-color: #FFFFFF;
              background-image: url('https://img.icons8.com/?size=100&id=132&format=png&color=000000');
              background-repeat: no-repeat;
              background-position: right 12px center;
              background-size: 20px 20px;
              font-size: 16px;
              color: #1B263B;
              transition: border-color 0.3s;
          }
          .stTextInput div div input:focus {
              border-color: #3498DB;
              outline: none;
          }
        
          .stButton>button {
              background-color: #3498DB;
              color: #FFFFFF;
              border-radius: 10px;
              padding: 12px 25px;
              border: none;
              transition: background-color 0.3s, transform 0.3s;
              font-size: 16px;
              font-weight: 500;
          }
          .stButton>button:hover {
              background-color: #2E86C1;
              transform: scale(1.05);
          }
        
          .stExpander {
              border: 2px solid #3498DB;
              border-radius: 10px;
              box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
              background-color: #FFFFFF;
              margin-top: 20px;
          }
          .stExpander div[role="button"] {
              background-color: #D6EAF8;
              color: #1B263B;
              font-weight: 600;
              padding: 10px;
              border-radius: 8px;
          }
        
          footer {
              margin-top: 30px;
              padding: 20px;
              text-align: center;
              border-top: 2px solid #D6EAF8;
              color: #1B263B;
              background-color: #FFFFFF;
              border-radius: 10px;
              box-shadow: 0 -2px 6px rgba(0, 0, 0, 0.1);
          }
        </style>
    """,
    unsafe_allow_html=True
)

# Logo & Title
st.logo(image="./logo.png", size='large')
st.title("Credit Risk Assessment Dashboard")

# Text input with on_change callback
def process_search():
    company_value = st.session_state.company_input.strip()
    if company_value:
        with st.spinner(f"Analyzing data for {company_value}... Please wait."):
            result = run_crew(company_value)
            if result:
                st.session_state['result'] = result
                st.session_state['company'] = company_value

st.text_input("", key="company_input", placeholder="Enter a company name and number e.g., Intel - 1094254", on_change=process_search)

# Display tabs if results exist
if 'result' in st.session_state:
    tabs = st.tabs(["Logs", "All", "Market Data", "Events", "Reviews", "Credit Risk","Decision", "Download Report"])
    comp = st.session_state.get('company', 'Unknown')
    result = st.session_state['result']

    with tabs[0]:
        if 'logs' in st.session_state:
            st.text_area("Agent Logs", st.session_state['logs'], height=300)
        else:
            st.text_area("Agent Logs", "No logs available", height=300)
    with tabs[1]:
        display_overview(result, key_suffix="all")
        display_market_data(result.tasks_output[0].output, key_suffix="all")
        display_events(result.tasks_output[1].output, key_suffix="all")
        display_reviews(result.tasks_output[2].output)
        display_credit_risk(result.tasks_output[3].output, key_suffix="all")
        display_strategic_decisions(result.tasks_output[4].output, key_suffix="all")
    # with tabs[2]:
    #     display_overview(result, key_suffix="overview")
    with tabs[2]:
        display_market_data(result.tasks_output[0].output, key_suffix="market")
    with tabs[3]:
        display_events(result.tasks_output[1].output, key_suffix="events")
    with tabs[4]:
        display_reviews(result.tasks_output[2].output)
    with tabs[5]:
        display_credit_risk(result.tasks_output[3].output, key_suffix="risk")
    with tabs[6]:
        display_strategic_decisions(result.tasks_output[4].output, key_suffix="strategic")  # Add this tab
    with tabs[7]:
        st.download_button(
            "Download PDF Report",
            data=generate_pdf_report(result, comp),
            file_name=f"{comp}_credit_risk_report.pdf",
            mime="application/pdf"
        )

st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    """
    <footer>
        <p>© 2025 DCodeAI | Credit Risk Dashboard v2.1.0</p>
    </footer>
    """,
    unsafe_allow_html=True
)