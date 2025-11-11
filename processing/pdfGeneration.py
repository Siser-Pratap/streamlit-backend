
# from processing import pdf_report_generator
from processing.pdf_report_generator import generate_pdf_report_from_excel
from processing.cloudinary import upload_to_cloudinary, delete_last_n_resources
from utils.logger import logger


import os
import re
import json
import datetime
from uuid import uuid4
import asyncio
import base64
from typing import List

from fastapi import HTTPException, UploadFile
from openai import OpenAI
import io

async def generate_pdf_report(file: UploadFile, reportTitle:str, brandName:str, includeInsights:bool, includeCharts:bool, customPrompt:str):
    """
    Generate a beautiful PDF report from Excel data with OpenAI insights.
    Controller function that orchestrates the PDF generation process using helper functions.
    """
    # Validate file type
    # if not file.filename.endswith(('.xlsx', '.xls')):
    #     raise HTTPException(
    #         status_code=400,
    #         detail="Invalid file format. Only Excel files (.xlsx, .xls) are supported."
    #     )
    
    try:
        # Delete last 5 resources from Cloudinary before generating new PDF
        logger.info("Deleting last 5 resources from Cloudinary before PDF generation...")
        try:
            deletion_result = delete_last_n_resources(n=5, resource_types=["image", "raw"])
            logger.info(f"Cloudinary cleanup completed: {deletion_result.get('total_deleted', 0)} deleted, {deletion_result.get('total_failed', 0)} failed")
        except Exception as cleanup_error:
            logger.warning(f"Cloudinary cleanup failed, continuing with PDF generation: {cleanup_error}")
        
        # Read Excel file
        excel_bytes = await file.read()
        
        # Use consolidated helper to generate insights and charts (charts saved in temp dir)
        pdf_buffer, insights, chart_urls = await generate_pdf_report_from_excel(
            excel_bytes,
            report_title=reportTitle,
            brand_name=brandName,
            custom_prompt=customPrompt
        )
        
        # Upload PDF to Cloudinary instead of saving locally
        pdf_bytes = pdf_buffer.getvalue()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"media_report_{timestamp}.pdf"
        
        try:
            # Upload PDF to Cloudinary (use resource_type="raw" for PDFs)
            cloudinary_pdf_url = upload_to_cloudinary(pdf_bytes, folder="reports", resource_type="raw")
            logger.info(f"PDF uploaded to Cloudinary: {cloudinary_pdf_url}")
        except Exception as e:
            logger.error(f"Failed to upload PDF to Cloudinary: {str(e)}")
            # Fallback to local save if Cloudinary fails
            pdf_path = os.path.join(config.UPLOAD_DIR, pdf_filename)
            os.makedirs(config.UPLOAD_DIR, exist_ok=True)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_bytes)
            cloudinary_pdf_url = f"/assets/{pdf_filename}"
            logger.warning(f"Cloudinary upload failed, saved locally: {pdf_path}")
        
        # Return response with chart URLs and Cloudinary PDF URL
        return {
        "success": True,
        "message": "PDF report generated successfully",
        "reportUrl": cloudinary_pdf_url,
        "insights": insights.get("insights", ""),
        "keyMessages": insights.get("key_messages", []),
        "suggestedChannels": insights.get("suggested_channels", []),
        "chartUrls": chart_urls
        }
        
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")