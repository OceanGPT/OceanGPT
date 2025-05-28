# OceanGPT MCP Server

This is an experimental Model Context Protocol (MCP) server implementation for OceanGPT.

- [OceanGPT MCP Server](#oceangpt-mcp-server)
  - [🌊 Features](#-features)
    - [Sonar Image Caption](#sonar-image-caption)
    - [Others](#others)
  - [🛰️ Server Deploy](#️-server-deploy)
    - [Local](#local)
    - [Remote](#remote)
  - [📡 Client Use](#-client-use)
    - [Cursor](#cursor)
    - [Claude](#claude)
    - [Cherry Studio](#cherry-studio)
    - [Others](#others-1)
  - [🏛️ License](#️-license)

## 🌊 Features

### Sonar Image Caption

Identify objects by using our Ocean Science Sonar Vision Model.

After **MCP Server** deployed, you can ask your LLM in your **MCP Client** like this:

```
Please use my MCP Server tool and answer:
What the sonar image ("https://raw.githubusercontent.com/zjunlp/OceanGPT/main/mcp_server/data/SonarImage002.png") returned by my marine detection robot means? What object could this image be?
```

> You can provide a **local** or **remote** image **path** when our MCP server is **local**.
>
> You can provide a **remote** image **path** when our MCP server is **remote**.



### Others

> Developing...



## 🛰️ Server Deploy

### Local

> Support both **STDIO** and **SSE** transport type.

1. **Clone the OceanGPT GitHub repo & Open this  MCP project**

```bash
git clone https://github.com/zjunlp/OceanGPT.git
cd OceanGPT/mcp_server
```

2. **Use `uv` to manage project (need install [uv](https://docs.astral.sh/uv/getting-started/installation/#installing-uv) first)**

Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv init
uv venv .venv --python 3.11
.venv\Scripts\activate.bat
uv pip install -r requirements.txt
```

MacOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

3. **Launch MCP local server**

You can choose STDIO (recommend) transport type or SSE (not recommend) transport type.

```bash
# choice 1: Use MCP Inspector (recommend)
fastmcp dev oceanserver.py
# You can easily test at http://127.0.0.1:6274

# choice 2: uv run
uv run --directory FULL_PATH/OceanGPT/mcp_server fastmcp run FULL_PATH/OceanGPT/mcp_server/oceanserver.py
# Note that the slash symbol for Win and Mac paths is different

# choice 3: python run SSE
python oceanserver_sse.py
# (Not recommend)
```

4. **Select your MCP client & add `json` config**

Please refer to the **next chapter** for details.

The `json` file usually look like:

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



### Remote

> Support **SSE** Transport Type, and please ensure that your LLM has access to **international** networks.

1. ***(Optional)* Test remote server by MCP Inspector**

```bash
# Make sure you already have MCP Inspector first
npx @modelcontextprotocol/inspector
# You can easily test at http://127.0.0.1:6274
```



2. **Select your MCP client & add `json` config (Without any other steps)**

Please refer to the **next chapter** for details.

The `json` file usually look like:

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



## 📡 Client Use

All these MCP client you just need modify their `json` config file.

### Cursor

> Support both **local** and **remote** server.

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

`FULL_PATH` is your local path.

**Remote json config:**

```json
{
  "mcpServers": {
    "OceanGPT_Remote": {
      "url": "OUR_REMOTE_URL",
    }
  }
}
```

`OUR_REMOTE_URL` currently is: https://oceangpt-mcp.onrender.com/sse



### Claude

> Only support **local** server now.

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

`FULL_PATH` is your local path.



### Cherry Studio

> Support both **local** and **remote** server, but please ensure that your LLM has access to **international** networks.

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

`FULL_PATH` is your local path.

**Remote json config:**

```json
{
  "mcpServers": {
    "YOUR_AUTO_GENERATED_HASH": {
      "name": "OceanGPT_Remote",
      "type": "sse",
      "description": "Please ensure that your LLM has access to international networks.",
      "isActive": true,
      "timeout": "10000",
      "baseUrl": "OUR_REMOTE_URL"
    }
  }
}
```

`OUR_REMOTE_URL` currently is: https://oceangpt-mcp.onrender.com/sse



### Others

> Developing...



## 🏛️ License

This MCP project is licensed under the same license as the parent full OceanGPT project. 🐱
