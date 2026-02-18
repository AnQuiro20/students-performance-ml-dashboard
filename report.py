from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io


def generate_pdf(best_model_name, metrics):
    """
    Generate PDF report in memory.
    """

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]

    elements.append(Paragraph("Students Performance ML Report", title_style))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(f"Best Model: {best_model_name}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Table data
    table_data = [["Model", "R²", "MAE", "RMSE", "CV R²"]]

    for model_name, m in metrics.items():
        table_data.append([
            model_name,
            f"{m['r2']:.3f}",
            f"{m['mae']:.2f}",
            f"{m['rmse']:.2f}",
            f"{m['cv_mean']:.3f}"
        ])

    table = Table(table_data)

    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph("Key Insights:", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    insights = """
    - Reading and writing scores strongly predict math performance.
    - Test preparation improves student outcomes.
    - Socioeconomic indicators influence academic results.
    - Model interpretability was validated using SHAP.
    """

    elements.append(Paragraph(insights, styles["Normal"]))

    doc.build(elements)

    buffer.seek(0)
    return buffer
