import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

class ChallanGenerator:
    def __init__(self, output_dir="data/challans"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, violation_data, image_path=None):
        """
        Generates a PDF e-challan.
        violation_data should be a dict with:
        - violation_id
        - timestamp
        - rule_name
        - location
        - speed_kmh (optional)
        - plate_number (optional)
        """
        challan_id = violation_data.get('violation_id', f"CH-{int(datetime.now().timestamp())}")
        filename = os.path.join(self.output_dir, f"{challan_id}.pdf")
        
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        
        # Header
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "TRAFFIC POLICE E-CHALLAN")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Challan No: {challan_id}")
        c.drawString(50, height - 100, f"Date & Time: {violation_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
        c.drawString(50, height - 120, f"Location: {violation_data.get('location', 'Camera 01')}")
        
        # Vehicle Details
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 160, "Vehicle Details")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 180, f"License Plate: {violation_data.get('plate_number', 'Not Detected')}")
        if 'speed_kmh' in violation_data and violation_data['speed_kmh'] > 0:
            c.drawString(50, height - 200, f"Recorded Speed: {violation_data['speed_kmh']:.1f} km/h")
            
        # Violation Details
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 240, "Violation Details")
        c.setFont("Helvetica", 12)
        rule_name = violation_data.get('rule_name', 'Unknown Violation')
        c.drawString(50, height - 260, f"Offence: {rule_name}")
        
        # Fine Amount (dummy logic)
        fine = 500
        if "helmet" in rule_name.lower() or "overcrowding" in rule_name.lower():
            fine = 1000
        elif "speed" in rule_name.lower():
            fine = 2000
        c.drawString(50, height - 280, f"Fine Amount: ₹ {fine}")
        
        # Evidence Image
        if image_path and os.path.exists(image_path):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 320, "Evidence Image:")
            try:
                img = ImageReader(image_path)
                # Draw image fitting in 400x300 box
                c.drawImage(img, 50, height - 640, width=400, height=300, preserveAspectRatio=True)
            except Exception as e:
                c.setFont("Helvetica", 10)
                c.drawString(50, height - 340, f"(Could not load image: {e})")

        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 50, "This is an automatically generated e-challan from IntelliTraffic AI Engine.")
        
        c.save()
        return filename
