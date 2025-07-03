# 🌊 OceanGPT MCP Server

English | [简体中文](https://github.com/zjunlp/OceanGPT/blob/main/mcp_server/README_CN.md)

This is an experimental Model Context Protocol ([MCP](https://modelcontextprotocol.io/introduction)) server implementation for our OceanGPT. Here is an example of the remote MCP server on Cursor:

<div align="center">
<img src="data/cursor_example.webp">
</div>



📖 **Contents**

- [🌊 OceanGPT MCP Server](#-oceangpt-mcp-server)
  - [🏄 Features](#-features)
    - [Sonar Image Caption](#sonar-image-caption)
    - [Fish Image Caption](#fish-image-caption)
    - [Others](#others)
  - [🛰️ Server Deploy](#️-server-deploy)
    - [Remote](#remote)
    - [Local](#local)
  - [📡 Client Host](#-client-host)
    - [Cursor](#cursor)
    - [Claude](#claude)
    - [Cherry Studio](#cherry-studio)
    - [Others](#others-1)
  - [🌻 Acknowledgement](#-acknowledgement)
  - [🏛️ License](#️-license)



## 🏄 Features

### Sonar Image Caption

Our Ocean Science Sonar Vision Model is designed to identify various underwater objects with high precision. It is capable of recognizing at least 10 types of underwater targets commonly encountered in ocean engineering and marine research scenarios, including the following categories:

| English Name | 中文名称 |
|:---:|:---:|
| Bottle       | 瓶状物    |
| Cube         | 立方体    |
| Cylinder     | 圆柱体    |
| Hook         | 钩状物    |
| Pipeline     | 管道     |
| Plane        | 飞机     |
| Propeller    | 螺旋桨    |
| Ship         | 船体     |
| Tire         | 轮胎     |
| Valve        | 阀门     |

After MCP Server deployed, you can ask your LLM just like this:

```
Please use my MCP tool to analyze the following sonar image:
["https://raw.githubusercontent.com/zjunlp/OceanGPT/main/mcp_server/data/SonarImage001.jpg"]
What does the image indicate, and what object might it represent?
```

**Make sure** your prompt contains at least a sonar image local path or remote url:

When our MCP server is local, you can provide a local path or remote url.

When our MCP server is remote, you can provide a remote url.

### Fish Image Caption

> Will be released soon...

### Others

> Will be released soon...



## 🛰️ Server Deploy

### Remote

> Support *SSE* Transport Type, and please ensure that your LLM can access international networks.

You actually only need to complete just one following step:

1. ***(Optional)* Test remote server by MCP Inspector**

```bash
# Make sure you already have MCP Inspector first
npx @modelcontextprotocol/inspector
# You can easily test at http://127.0.0.1:6274
```



2. **Select your MCP client & add `json` config (Without any other steps)**

The `json` file may look like:

```json
{
  "mcpServers": {
    "OceanGPT": {
      "url": "https://.../sse",
      "env": {
        "YOUR_API_KEY": "..."
      }
    }
  }
}
```

<a id="current_url"></a>

👇 The current remote server URL is:

```
https://oceangpt-mcp.onrender.com/sse
```

or:

```
https://oceangpy-c-cite-mwlqnwjval.cn-hangzhou.fcapp.run/sse
```

👆

> [!NOTE]
> It may time-out due to network connection. So if this happens many times, we recommend using the local MCP service mentioned below.



### Local

> Support both *STDIO* and *SSE* transport type.

Just follow the following steps:

1. **Clone the OceanGPT GitHub repo & Open this  MCP project**

```bash
git clone https://github.com/zjunlp/OceanGPT.git
cd OceanGPT/mcp_server
```

2. **Use `uv` to manage project (need to [install uv](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) first)**

**Windows:**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv init
uv venv .venv --python 3.11
.venv\Scripts\activate.bat
uv pip install -r requirements.txt
```

If your environment is ready, just activate it.

```bash
.venv\Scripts\activate.bat
```

**MacOS or Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

If your environment is ready, just activate it.

```bash
source .venv/bin/activate
```

3. **Launch MCP local server**

You can choose *STDIO **(recommend Option-1)** transport type or *SSE* (not recommend) transport type.

```bash
# Option-1: Use MCP Inspector (recommended)
# Make sure not to be bothered by the path.
.\.venv\Scripts\fastmcp.exe dev oceanserver.py
# or: fastmcp dev oceanserver.py
# You can easily test at http://127.0.0.1:6274

# Option-2: uv run
uv run --directory FULL_PATH/OceanGPT/mcp_server fastmcp run FULL_PATH/OceanGPT/mcp_server/oceanserver.py
# Note that the slash symbol for Win and Mac path is different

# Option-3: python run with SSE
# Modify oceanserver.py:
# mcp.run() --> mcp.run(transport="sse")
python oceanserver.py
# The default local SSE URL at http://127.0.0.1:8000
```

4. **Select your MCP client & add `json` config**

Please refer to [the next chapter](#-client-host) for details.

The `json` file may look like:

```json
{
  "mcpServers": {
    "OceanGPT": {
      "command": "python/uv",
      "args": [
          "...",
          "..."
      ]
    }
  }
}
```



## 📡 Client Host

All these MCP client you just need modify their `json` config file.

### Cursor

> Support both *local* and *remote* server.

**Local json config:**

```json
{
  "mcpServers": {
    "OceanGPT_Local": {
      "command": "uv",
      "args": [
          "run",
          "--directory",
          "FULL_PATH/OceanGPT/mcp_server",
          "fastmcp",
          "run",
          "FULL_PATH/OceanGPT/mcp_server/oceanserver.py"
      ]
    }
  }
}
```

`FULL_PATH` is your local path. Note that the slash symbol for Win and Mac path is different. Make sure not to be bothered by the path.

**Remote json config:**

```json
{
  "mcpServers": {
    "OceanGPT_Remote": {
      "url": "OUR_REMOTE_URL"
    }
  }
}
```

`OUR_REMOTE_URL` is [here](#current_url).



### Claude

> Only support *local* server now.

**Local json config:**

```json
{
  "mcpServers": {
    "OceanGPT": {
      "command": "uv",
      "args": [
          "run",
          "--directory",
          "FULL_PATH/OceanGPT/mcp_server",
          "fastmcp",
          "run",
          "FULL_PATH/OceanGPT/mcp_server/oceanserver.py"
      ]
    }
  }
}
```

`FULL_PATH` is your local path. Note that the slash symbol for Win and Mac path is different. Make sure not to be bothered by the path.



### Cherry Studio

> Support both *local* and *remote* server, but please ensure that your LLM can access international networks.

**Local json config:**

```json
{
  "mcpServers": {
    "YOUR_AUTO_GENERATED_HASH": {
      "name": "OceanGPT_Local",
      "type": "stdio",
      "description": "YOUR_DESCRIPTION",
      "isActive": true,
      "registryUrl": "https://pypi.tuna.tsinghua.edu.cn/simple",
      "timeout": "10000",
      "tags": [],
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "FULL_PATH/OceanGPT/mcp_server",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "FULL_PATH/OceanGPT/mcp_server/oceanserver.py"
      ]
    }
  }
}
```

`FULL_PATH` is your local path. Note that the slash symbol for Win and Mac path is different. Make sure not to be bothered by the path.

**Remote json config:**

```json
{
  "mcpServers": {
    "YOUR_AUTO_GENERATED_HASH": {
      "name": "OceanGPT_Remote",
      "type": "sse",
      "description": "Please ensure that your LLM can access international networks.",
      "isActive": true,
      "timeout": "10000",
      "baseUrl": "OUR_REMOTE_URL"
    }
  }
}
```

`OUR_REMOTE_URL` is [here](#current_url).



### Others

> Welcome your suggestions and additions.



## 🌻 Acknowledgement

OceanGPT (沧渊) is trained based on the open-sourced large language models including [Qwen](https://huggingface.co/Qwen), [MiniCPM](https://huggingface.co/collections/openbmb/minicpm-2b-65d48bf958302b9fd25b698f), [LLaMA](https://huggingface.co/meta-llama).

OceanGPT is trained based on the open-sourced data and tools including [Moos](https://github.com/moos-tutorials), [UATD](https://openi.pcl.ac.cn/OpenOrcinus_orca/URPC2021_sonar_images_dataset), [Forward-looking Sonar Detection Dataset](https://github.com/XingYZhu/Forward-looking-Sonar-Detection-Dataset), [NKSID](https://github.com/Jorwnpay/NK-Sonar-Image-Dataset), [SeabedObjects-KLSG](https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset), [Marine Debris](https://github.com/mvaldenegro/marine-debris-fls-datasets/tree/master/md_fls_dataset/data/turntable-cropped).

Thanks for their great contributions!



## 🏛️ License

This MCP project is licensed under the same license as the parent full OceanGPT project.

🐱
