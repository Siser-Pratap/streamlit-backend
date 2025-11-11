from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI

from utils.logger import logger
import config
import os

from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _safe_read_all_sheets(excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    try:
        return pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
    except Exception as exc:
        logger.error(f"Unable to read processed Excel: {exc}")
        return {}


def extract_from_data_chart(df: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {"sentiment": [], "type": []}
    if df is None or df.empty:
        return result

    # Expect sentiment table at columns A:B and type table at E:F from generator
    try:
        # Sentiment (first two columns)
        sentiment_df = (
            df.iloc[:, 0:2]
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )
        # Normalize headers if available
        if len(sentiment_df.columns) == 2:
            sentiment_df.columns = ["label", "count"]
            sentiment_records = (
                sentiment_df.dropna(how="any")
                .assign(count=lambda x: pd.to_numeric(x["count"], errors="coerce").fillna(0).astype(int))
                .to_dict("records")
            )
            result["sentiment"] = sentiment_records
    except Exception as exc:
        logger.warning(f"Failed parsing sentiment table: {exc}")

    try:
        # Type (two columns starting at index 4)
        if df.shape[1] >= 6:
            type_df = (
                df.iloc[:, 4:6]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            if len(type_df.columns) == 2:
                type_df.columns = ["label", "count"]
                type_records = (
                    type_df.dropna(how="any")
                    .assign(count=lambda x: pd.to_numeric(x["count"], errors="coerce").fillna(0).astype(int))
                    .to_dict("records")
                )
                result["type"] = type_records
    except Exception as exc:
        logger.warning(f"Failed parsing type table: {exc}")

    return result


def extract_from_summary(df: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {"totals": [], "insights": []}
    if df is None or df.empty:
        return result

    # Try to pick totals if there are "Metric" and "Value" columns anywhere
    try:
        summary_header_df = pd.read_excel(io.BytesIO(df.to_excel(index=False)), header=0)
    except Exception:
        summary_header_df = df.copy()

    try:
        if {"Metric", "Value"}.issubset(set(summary_header_df.columns)):
            totals = (
                summary_header_df[["Metric", "Value"]]
                .dropna(how="any")
                .to_dict("records")
            )
            result["totals"] = totals
    except Exception as exc:
        logger.warning(f"Failed extracting totals from summary: {exc}")

    # Fallback: look for an "Insights" column
    try:
        possible = [col for col in summary_header_df.columns if str(col).strip().lower() == "insights"]
        if possible:
            insights_series = summary_header_df[possible[0]].dropna()
            result["insights"] = [str(x).strip() for x in insights_series.tolist() if str(x).strip()]
    except Exception as exc:
        logger.warning(f"Failed extracting insights column: {exc}")

    return result


def prepare_prompt(data_chart: Dict[str, Any], summary: Dict[str, Any], report_title: str, brand_name: str, custom_prompt: str | None) -> str:
    lines: List[str] = []
    lines.append(f"REPORT: {report_title}")
    lines.append(f"BRAND: {brand_name}")

    lines.append("\nDATA_CHART - SENTIMENT:")
    for row in data_chart.get("sentiment", [])[:10]:
        lines.append(f"- {row.get('label')}: {row.get('count')}")

    lines.append("\nDATA_CHART - TYPE:")
    for row in data_chart.get("type", [])[:10]:
        lines.append(f"- {row.get('label')}: {row.get('count')}")

    lines.append("\nSUMMARY TOTALS:")
    for row in summary.get("totals", [])[:10]:
        lines.append(f"- {row.get('Metric')}: {row.get('Value')}")

    if summary.get("insights"):
        lines.append("\nEXISTING INSIGHTS:")
        for it in summary["insights"][:10]:
            lines.append(f"- {it}")

    base = "\n".join(lines)

    prompt = f"""
    Analyze the following media monitoring data and provide comprehensive insights:

    DATA:
    {summary}

    Please provide:
    1. Executive Summary (2-3 paragraphs)
    2. Key Performance Indicators Analysis
    3. Trend Analysis
    4. Brand Performance Insights
    5. Competitive Landscape Observations
    6. Recommendations for Future Actions
    7. Top 5 Key Messages
    8. Top 5 Suggested Communication Channels

    Brand Context: {brand_name}
    Report Focus: {report_title}
    """
    if custom_prompt:
        prompt += f"\nAdditional Context: {custom_prompt}"

    return prompt


def parse_insights_response(insights_text: str) -> Dict[str, Any]:
    try:
        result: Dict[str, Any] = {
            "insights": insights_text,
            "key_messages": [],
            "suggested_channels": [],
        }

        if "KEY MESSAGES:" in insights_text:
            section = insights_text.split("KEY MESSAGES:")[1]
            if "SUGGESTED CHANNELS:" in section:
                section = section.split("SUGGESTED CHANNELS:")[0]
            lines = [ln.strip() for ln in section.split("\n") if ln.strip()]
            msgs: List[str] = []
            for ln in lines:
                cleaned = ln.lstrip("-* ").strip()
                if cleaned:
                    msgs.append(cleaned)
            result["key_messages"] = msgs[:5]

        if "SUGGESTED CHANNELS:" in insights_text:
            section = insights_text.split("SUGGESTED CHANNELS:")[1]
            lines = [ln.strip() for ln in section.split("\n") if ln.strip()]
            chans: List[str] = []
            for ln in lines:
                cleaned = ln.lstrip("-* ").strip()
                if cleaned:
                    chans.append(cleaned)
            result["suggested_channels"] = chans[:5]

        return result
    except Exception as exc:
        logger.error(f"Error parsing insights: {exc}")
        return {"insights": insights_text, "key_messages": [], "suggested_channels": []}


# ---------------------- Simple PDF builder ----------------------
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


def _build_simple_pdf(report_title: str, brand_name: str, data_chart: Dict[str, Any], summary: Dict[str, Any], insights: Dict[str, Any]) -> io.BytesIO:
    buffer = io.BytesIO()
    if not REPORTLAB_AVAILABLE:
        # Return minimal buffer if reportlab missing
        buffer.write(b"PDF generation not available.")
        buffer.seek(0)
        return buffer

    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=48, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: List[Any] = []

    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], alignment=1, textColor=colors.HexColor('#1F4E78')
    )
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Brand: {brand_name}", styles['Normal']))
    story.append(Spacer(1, 18))

    # Executive summary
    if insights.get('insights'):
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        for para in str(insights['insights']).split('\n\n')[:5]:
            if para.strip():
                story.append(Paragraph(para.strip(), styles['Normal']))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

    # Key Messages
    if insights.get('key_messages'):
        story.append(Paragraph("Key Messages", styles['Heading2']))
        for i, msg in enumerate(insights['key_messages'][:5], 1):
            story.append(Paragraph(f"{i}. {msg}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Suggested Channels
    if insights.get('suggested_channels'):
        story.append(Paragraph("Suggested Channels", styles['Heading2']))
        for i, ch in enumerate(insights['suggested_channels'][:5], 1):
            story.append(Paragraph(f"{i}. {ch}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Summary totals table
    totals = summary.get('totals', [])
    if totals:
        story.append(Paragraph("Summary Totals", styles['Heading2']))
        data = [["Metric", "Value"]] + [[t.get('Metric', ''), str(t.get('Value', ''))] for t in totals]
        tbl = Table(data, hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F0F4F8')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1F4E78')),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))

    # Data chart summaries
    if data_chart.get('sentiment') or data_chart.get('type'):
        story.append(Paragraph("Data Chart Summary", styles['Heading2']))
        if data_chart.get('sentiment'):
            story.append(Paragraph("Sentiment Counts", styles['Heading3']))
            data = [["Label", "Count"]] + [[r.get('label', ''), str(r.get('count', ''))] for r in data_chart['sentiment']]
            tbl = Table(data, hAlign='LEFT')
            tbl.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey)]))
            story.append(tbl)
            story.append(Spacer(1, 8))
        if data_chart.get('type'):
            story.append(Paragraph("Type Counts", styles['Heading3']))
            data = [["Label", "Count"]] + [[r.get('label', ''), str(r.get('count', ''))] for r in data_chart['type']]
            tbl = Table(data, hAlign='LEFT')
            tbl.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey)]))
            story.append(tbl)

    doc.build(story)
    buffer.seek(0)
    return buffer


async def generate_insights_from_processed_excel(excel_bytes: bytes, report_title: str, brand_name: str, custom_prompt: str | None = None) -> Tuple[io.BytesIO, Dict[str, Any]]:
    sheets = _safe_read_all_sheets(excel_bytes)
    if not sheets:
        empty_pdf = _build_simple_pdf(report_title, brand_name, {}, {}, {"insights": "No data found in file.", "key_messages": [], "suggested_channels": []})
        return empty_pdf, {"insights": "No data found in file.", "key_messages": [], "suggested_channels": []}

    data_chart_df = sheets.get("data_chart")
    summary_df = sheets.get("summary")

    data_chart = extract_from_data_chart(data_chart_df) if data_chart_df is not None else {"sentiment": [], "type": []}
    summary = extract_from_summary(summary_df) if summary_df is not None else {"totals": [], "insights": []}

    prompt = prepare_prompt(data_chart, summary, report_title, brand_name, custom_prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert media analyst. Be concise and actionable."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        text = response.choices[0].message.content
        insights = parse_insights_response(text)
        pdf_buffer = _build_simple_pdf(report_title, brand_name, data_chart, summary, insights)
        return pdf_buffer, insights
    except Exception as exc:
        logger.error(f"OpenAI error: {exc}")
        insights = {"insights": "Unable to generate insights.", "key_messages": [], "suggested_channels": []}
        pdf_buffer = _build_simple_pdf(report_title, brand_name, data_chart, summary, insights)
        return pdf_buffer, insights


