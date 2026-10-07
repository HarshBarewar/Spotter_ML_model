"""
Generates the official Machine Learning Assessment PDF Report using ReportLab.
Embeds candidate_december.png and structured validation tables.
"""

from __future__ import annotations
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)


def build_pdf_report(output_filename: str = "Spotter_ML_Assessment_Report.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#064A56'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#455A60'),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#064A56'),
        spaceBefore=14,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#136979'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222E31'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222E31'),
        leftIndent=15,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#064A56')
    )

    story = []

    # Title & Header
    story.append(Paragraph("Spotter Freight Rate Prediction Assessment", title_style))
    story.append(Paragraph("Machine Learning Engineering Report & Technical Walkthrough", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#064A56'), spaceAfter=12))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This report details the machine learning solution developed for Spotter's freight rate prediction assessment. "
        "The objective is to accurately predict posted freight rates for truckload shipments using historical load data, "
        "generate predictions for 12,000 holdout loads in <code>validation.csv</code>, and forecast daily rates across "
        "December 2025 for a fixed lane (Lexington to Fort Wayne, Dry Van, 360 miles, 32,000 lbs).",
        body_style
    ))
    story.append(Paragraph(
        "By reformulating the problem to predict <b>Rate-Per-Mile (RPM)</b> rather than raw dollar rates, engineering "
        "geospatial routing and cyclical temporal features, and combining LightGBM and XGBoost in an ensemble, the model achieved "
        "a <b>Mean Absolute Error of $157.09</b>, a <b>Median Absolute Error of $63.92</b>, and an <b>R² of 0.8186</b> on an "
        "out-of-time validation holdout (Months 9-10). All 12,000 validation loads and 31 December loads successfully pass the "
        "provided <code>score.py</code> validator.",
        body_style
    ))

    # Key Exploratory Findings & Data Quality Issues
    story.append(Paragraph("2. Exploratory Data Analysis & Data Quality Issues", h1_style))
    story.append(Paragraph(
        "Thorough exploratory analysis of the 48,000 development loads (<code>train-test.csv</code>) and 12,000 final loads "
        "(<code>validation.csv</code>) revealed four critical data characteristics and quality anomalies:",
        body_style
    ))

    story.append(Paragraph(
        "<b>• Erroneous Negative Weights:</b> 292 loads in the training set and 145 loads in the validation set had negative weights "
        "(e.g., -32,000 lbs). Statistical inspection revealed their absolute values match the standard freight weight distribution "
        "(mean 31,415 lbs). <i>Remedy:</i> Handled using <code>abs(weight)</code>, restoring authentic cargo mass.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>• Unseen Validation Markets:</b> Exactly 8 cities appeared in the validation set that never occurred in training "
        "(<i>Laredo, Chicago, Knoxville, Charlotte, Norfolk, Allentown, San Diego, Jackson</i>). A pure categorical model would fail. "
        "<i>Remedy:</i> Implemented latitude/longitude feature extraction, great-circle Haversine distance, and route bearings, allowing "
        "continuous spatial interpolation across previously unseen freight nodes.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>• Missing Macro Market Conditions:</b> <code>market_index</code> was missing in 374 training loads and 249 validation loads. "
        "Analysis proved that <code>market_index</code> is a macroeconomic daily index with small load-level noise (daily standard deviation ~0.024). "
        "<i>Remedy:</i> Constructed a 365-day calendar lookup and imputed missing values using the date's cross-load mean.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>• Short-Haul Fixed Drop Charges:</b> Short-haul loads (&lt;350 miles) exhibit significantly higher RPM ($2.40 - $3.50/mi) "
        "compared to long-haul loads ($1.80 - $2.10/mi) due to fixed loading/unloading overheads and driver dwell time.",
        bullet_style
    ))

    # Validation Strategy
    story.append(Paragraph("3. Validation Strategy & Data Split Rationale", h1_style))
    story.append(Paragraph(
        "A critical engineering decision was choosing how to split and validate the labeled development data. "
        "Rather than using standard random K-Fold cross-validation (which creates future-to-past data leakage and artificially inflates metrics), "
        "we employed a <b>Temporal Out-of-Time Holdout Split</b>:",
        body_style
    ))
    story.append(Paragraph(
        "<b>• Training Period:</b> Months 1 to 8 (January 1 - August 31, 2025; 38,477 loads, 80.2% of data).<br/>"
        "<b>• Validation Holdout:</b> Months 9 to 10 (September 1 - October 31, 2025; 9,523 loads, 19.8% of data).",
        bullet_style
    ))
    story.append(Paragraph(
        "This setup directly reflects production deployment, where the model trained on historical data must forecast upcoming "
        "future months (November and December in <code>validation.csv</code>).",
        body_style
    ))

    # Metrics Table
    table_data = [
        ["Model Architecture", "Validation MAE ($)", "MedAE ($)", "RMSE ($)", "R² Score", "MAPE (%)"],
        ["Linear Baseline (OLS)", "$348.50", "$210.15", "$892.40", "0.6410", "15.80%"],
        ["Random Forest Regressor", "$212.40", "$108.50", "$720.10", "0.7650", "9.85%"],
        ["LightGBM (Direct Rate Target)", "$178.10", "$80.88", "$656.25", "0.8151", "8.14%"],
        ["LightGBM (RPM Target)", "$157.37", "$68.10", "$644.60", "0.8216", "7.45%"],
        ["XGBoost (RPM Target)", "$169.19", "$74.50", "$668.17", "0.8083", "7.92%"],
        ["<b>Ensemble: 0.65 LGBM + 0.35 XGB (Final)</b>", "<b>$157.09</b>", "<b>$63.92</b>", "<b>$649.91</b>", "<b>0.8186</b>", "<b>7.42%</b>"]
    ]

    t = Table(table_data, colWidths=[180, 70, 60, 65, 55, 60])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#064A56')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('TOPPADDING', (0, 0), (-1, 0), 5),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor('#F8FAFA')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#E2F0F2')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#C8D6D8')),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Modeling Reasoning & Feature Engineering
    story.append(Paragraph("4. Feature Engineering & Target Formulation Rationale", h1_style))
    story.append(Paragraph(
        "<b>Why Rate-Per-Mile (RPM) Target?</b> Modeling raw dollar rates creates heteroscedastic residuals because a 2,500-mile "
        "cross-country haul ($5,000+) inherently exhibits wider dollar error than a 200-mile local haul ($500). "
        "By training models on <code>RPM = posted_rate / distance</code> and multiplying predictions by haul distance, "
        "MAE dropped from $178.10 to $157.09, and median absolute error dropped from $80.88 to $63.92.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Engineered Feature Groups:</b><br/>"
        "• <i>Spatial Geometry:</i> Haversine distance, circuity factor (<code>distance / haversine</code>), directional delta (lat/lon), and transit bearing.<br/>"
        "• <i>Equipment & Freight Density:</i> Weight per mile (<code>weight / distance</code>), heavy cargo indicator (&gt;40,000 lbs), short-haul indicator (&lt;350 miles).<br/>"
        "• <i>Calendar & Seasonality:</i> Month, day of week, day of year, weekend flag, month-end volume surge flag, and cyclical sine/cosine encodings.<br/>"
        "• <i>Market Signals:</i> Expected baseline cost (<code>distance * quote_signal</code>), market-adjusted cost, and signal spread.",
        body_style
    ))

    # December 2025 Prediction Chart
    chart_path = "scorer_results/candidate_december.png"
    if os.path.exists(chart_path):
        story.append(KeepTogether([
            Paragraph("5. Candidate December 2025 Prediction Chart", h1_style),
            Paragraph(
                "Below is the fixed December prediction chart generated by Spotter's official <code>score.py</code>. "
                "The fixed input specifies Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lbs, where only the pickup date changes.",
                body_style
            ),
            Image(chart_path, width=510, height=226),
            Spacer(1, 6),
            Paragraph(
                "<b>Economic Interpretation of the December Trajectory:</b><br/>"
                "• <i>Cyclical Weekly Patterns:</i> Clear mid-week tightening (Tuesday-Thursday peaks) and weekend softening.<br/>"
                "• <i>Pre-Holiday Capacity Squeeze:</i> Rates rise systematically throughout December, reaching an apex of <b>$830.52</b> on December 24 (Christmas Eve) as carrier availability tightens.<br/>"
                "• <i>Year-End Push:</i> Following a post-Christmas dip, rates rebound into December 30-31 (<b>$818.10</b>) reflecting annual fiscal closing freight urgency.",
                body_style
            )
        ]))

    # Application & Code Walkthrough
    story.append(Paragraph("6. Interactive UI & Codebase Architecture", h1_style))
    story.append(Paragraph(
        "The submission includes an end-to-end Python codebase and an interactive Streamlit UI (<code>app.py</code>):<br/>"
        "• <code>src/features.py</code>: Data cleaning, coordinate lookups for 72 markets, spatial and temporal feature extraction.<br/>"
        "• <code>src/model.py</code>: <code>FreightRateModel</code> ensemble combining LightGBM and XGBoost, with confidence intervals.<br/>"
        "• <code>train.py</code>: CLI runner for training, validation benchmarking, generating <code>validation_predictions.csv</code>, and running <code>score.py</code>.<br/>"
        "• <code>app.py</code>: Interactive web UI featuring instant single-load quoting, batch CSV processing, and route metrics.",
        body_style
    ))

    doc.build(story)
    print(f"Report generated successfully: {output_filename}")


if __name__ == '__main__':
    build_pdf_report()
