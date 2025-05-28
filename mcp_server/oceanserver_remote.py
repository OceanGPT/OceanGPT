# Remote by SSE
from starlette.applications import Starlette
from starlette.routing import Mount, Host
from mcp.server.fastmcp import FastMCP
import uvicorn
import aiohttp

import os
import requests
from io import BytesIO


mcp = FastMCP("OceanGPT-MCP-Remote")

@mcp.tool()
def getSonarClassify(image_path_or_url: str) -> str:
    """
    Classify sonar image by using professional Ocean Science sonar image classification model.
    Supports both local file paths and remote image URLs.
    """
    url = os.environ["SONAR_CLASSIFY_URL"]

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


app = Starlette(
    routes=[
        Mount('/', app=mcp.sse_app()),
    ]
)
