# pdf_report.py
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from datetime import datetime
def generate_pdf(dataframe):
    """Verilerden PDF raporu oluştur"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    style = getSampleStyleSheet()

    # Rapor başlığı
    title = Paragraph("Türkiye Depremleri Raporu", style["Title"])
    elements.append(title)

    # Rapor bilgileri
    info_text = f"""
    <para align=right spaceb=3>
    <font size=10>
    <b>Rapor Tarihi:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")}<br/>
    <b>Toplam Deprem Sayısı:</b> {len(dataframe)}<br/>
    <b>En Şiddetli Deprem:</b> {dataframe['magnitude'].max() if not dataframe.empty else 'N/A'}<br/>
    <b>En Son Deprem:</b> {dataframe['time'].max().date() if not dataframe.empty else 'N/A'}
    </font>
    </para>
    """
    elements.append(Paragraph(info_text, style["Normal"]))

    # Ana tablo
    data = [list(dataframe.columns)]
    for row in dataframe.values:
        data.append([str(item) for item in row])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),  # Başlık satırı arka plan rengi
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),  # Başlık metin rengi
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),  # Tüm sütunları ortala
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),  # Başlık fontu
        ("FONTSIZE", (0, 0), (-1, 0), 10),  # Başlık font boyutu
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),  # Başlık altındaki boşluk
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),  # Veri satırı arka plan rengi
        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),  # Izgara çizgisi rengi
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),  # Veri satırı fontu
        ("FONTSIZE", (0, 1), (-1, -1), 8),  # Veri satırı font boyutu
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)  
    return buffer
