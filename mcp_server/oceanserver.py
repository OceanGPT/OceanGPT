# Local by STDIO
import os
import requests
from fastmcp import FastMCP
from io import BytesIO


mcp = FastMCP("OceanGPT-MCP-Local")
mcp.dependencies = []

@mcp.tool()
def getSonarClassify(image_path_or_url: str) -> str:
    """
    Classify sonar image by using professional Ocean Science sonar image classification model.
    Supports both local file paths and remote image URLs.
    """
    url = os.environ.get("SONAR_MODEL_URL", "http://121.41.117.246:8844/classify")

    try:
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            # remote image
            img_response = requests.get(image_path_or_url)
            img_response.raise_for_status()
            files = {"file": ("image.jpg", BytesIO(img_response.content))}
        else:
            # local image
            with open(image_path_or_url, "rb") as f:
                files = {"file": f}

        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.text

    except Exception as e:
        return f"Error: {e}"

# @mcp.tool()
# def otherTool():
#     pass


if __name__ == "__main__":
    mcp.run()
