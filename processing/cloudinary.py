import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
import os
from fastapi import HTTPException
from utils.logger import logger

load_dotenv()

cloudinary.config( 
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.getenv("CLOUDINARY_API_KEY"), 
    api_secret = os.getenv("CLOUDINARY_API_SECRET"), # Click 'View API Keys' above to copy your API secret
    secure=True
)

def upload_to_cloudinary(file_bytes, folder="uploads", resource_type="auto"):
    """
    Upload file to Cloudinary.
    
    Args:
        file_bytes: Bytes of the file to upload
        folder: Folder path in Cloudinary
        resource_type: Type of resource - "auto" (detects automatically), "image", "raw" (for PDFs)
    
    Returns:
        str: Secure URL of the uploaded file
    """
    try:
        # Auto-detect resource type if not specified
        if resource_type == "auto":
            # Check if it's a PDF by checking first bytes
            if file_bytes[:4] == b'%PDF':
                resource_type = "raw"
            else:
                resource_type = "image"
        
        upload_result = cloudinary.uploader.upload(
            file_bytes,
            folder=folder,
            resource_type=resource_type,
            use_filename=True,
            unique_filename=True
        )
        return upload_result.get("secure_url")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloudinary upload failed: {str(e)}")


def delete_from_cloudinary(file_url: str):
    """
    Delete a file (image or PDF) from Cloudinary using its URL.
    
    Args:
        file_url: The Cloudinary secure URL of the file to delete
    
    Returns:
        dict: Result of the deletion operation
    """
    try:
        # Extract public_id from Cloudinary URL
        # Cloudinary URLs format: 
        # Images: https://res.cloudinary.com/{cloud_name}/image/upload/{folder}/{public_id}.{format}
        # PDFs: https://res.cloudinary.com/{cloud_name}/raw/upload/{folder}/{public_id}.{format}
        # or with version: /upload/v{version}/{folder}/{public_id}.{format}
        import re
        from urllib.parse import urlparse
        
        # Parse the URL
        parsed = urlparse(file_url)
        path = parsed.path
        
        # Determine resource type from path
        if '/image/upload' in path:
            resource_type = "image"
            # Pattern: /image/upload/v{version}/{folder}/{public_id}.{format}
            # or: /image/upload/{folder}/{public_id}.{format}
            match = re.search(r'/image/upload(?:/v\d+)?/(.+?)(?:\.\w+)?$', path)
        elif '/raw/upload' in path:
            resource_type = "raw"
            # Pattern: /raw/upload/v{version}/{folder}/{public_id}.{format}
            # or: /raw/upload/{folder}/{public_id}.{format}
            match = re.search(r'/raw/upload(?:/v\d+)?/(.+?)(?:\.\w+)?$', path)
        else:
            raise ValueError(f"Unsupported Cloudinary resource type in URL: {file_url}")
        
        if not match:
            raise ValueError(f"Invalid Cloudinary URL format: {file_url}")
        
        public_id = match.group(1)
        
        # Delete the file
        result = cloudinary.uploader.destroy(
            public_id,
            resource_type=resource_type
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloudinary deletion failed: {str(e)}")


def delete_last_n_resources(n: int = 5, resource_types: list = None):
    """
    Delete the last N resources from Cloudinary (sorted by creation date, newest first).
    
    Args:
        n: Number of resources to delete (default: 5)
        resource_types: List of resource types to search for. If None, searches both "image" and "raw"
    
    Returns:
        dict: Summary of deletion results with 'deleted', 'failed', and 'total' counts
    """
    if resource_types is None:
        resource_types = ["image", "raw"]
    
    deleted = []
    failed = []
    
    try:
        for resource_type in resource_types:
            try:
                # List resources using Admin API
                # Get more resources than needed to ensure we have enough
                resources_result = cloudinary.api.resources(
                    type="upload",
                    resource_type=resource_type,
                    max_results=n * 2  # Get more to ensure we have enough after sorting
                )
                
                resources = resources_result.get("resources", [])
                
                # Sort by creation date (newest first) - created_at is in format "2024-01-01T12:00:00Z"
                resources_sorted = sorted(
                    resources,
                    key=lambda x: x.get("created_at", ""),
                    reverse=True
                )
                
                # Delete the first n resources (newest ones)
                for resource in resources_sorted[:n]:
                    public_id = resource.get("public_id")
                    if public_id:
                        try:
                            result = cloudinary.uploader.destroy(
                                public_id,
                                resource_type=resource_type
                            )
                            if result.get("result") == "ok":
                                deleted.append({
                                    "public_id": public_id,
                                    "resource_type": resource_type,
                                    "url": resource.get("secure_url", "")
                                })
                                logger.info(f"Deleted {resource_type} resource: {public_id}")
                            else:
                                failed.append({
                                    "public_id": public_id,
                                    "resource_type": resource_type,
                                    "error": result.get("result", "unknown error")
                                })
                        except Exception as e:
                            failed.append({
                                "public_id": public_id,
                                "resource_type": resource_type,
                                "error": str(e)
                            })
                            logger.error(f"Failed to delete {resource_type} resource {public_id}: {e}")
                            
            except Exception as e:
                logger.error(f"Error listing {resource_type} resources: {e}")
                # Continue with other resource types even if one fails
        
        return {
            "deleted": deleted,
            "failed": failed,
            "total_deleted": len(deleted),
            "total_failed": len(failed)
        }
    except Exception as e:
        logger.error(f"Error in delete_last_n_resources: {e}")
        return {
            "deleted": deleted,
            "failed": failed,
            "total_deleted": len(deleted),
            "total_failed": len(failed),
            "error": str(e)
        }