# OceanGPT MCP 用户手册

[English](https://github.com/zjunlp/OceanGPT/blob/main/mcp_server/README.md) | 简体中文


> [!NOTE]
> 本中文文档为简要版，一切以[完整文档](https://github.com/zjunlp/OceanGPT/blob/main/mcp_server/README.md)为最新。



**目录**
- [OceanGPT MCP 用户手册](#oceangpt-mcp-用户手册)
  - [服务端部署](#服务端部署)
    - [远程部署](#远程部署)
    - [本地部署](#本地部署)
  - [客户端调用](#客户端调用)



## 服务端部署

### 远程部署

> 支持 SSE 传输方式，须确保你的大模型具备访问国际网络的能力。

实际上只需完成这一个步骤：

1. **（可选）使用 MCP Inspector 测试远程服务器**

```bash
# 请确保已安装 MCP Inspector
npx @modelcontextprotocol/inspector
# 可在 http://127.0.0.1:6274 进行测试
```

1. **选择 MCP 客户端并添加 `json` 配置（无需其他额外操作）**

具体操作请参见下一章节。（客户端使用）

配置文件示例：

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



### 本地部署

> 支持 STDIO 与 SSE 两种传输方式。

只需按照以下步骤操作：

1. **克隆 OceanGPT 仓库并进入 MCP 项目目录**

```bash
git clone https://github.com/zjunlp/OceanGPT.git
cd OceanGPT/mcp_server
```

1. **使用 `uv` 管理项目（需先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/#installing-uv)）**

**Windows 系统：**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv init
uv venv .venv --python 3.11
.venv\Scripts\activate.bat
uv pip install -r requirements.txt
```

如果环境已准备好，直接激活即可：

```bash
.venv\\Scripts\\activate.bat
```

**MacOS 或 Linux 系统：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

如果环境已准备好，直接激活即可：

```bash
source .venv/bin/activate
```

1. **启动 MCP 本地服务器**

你可以选择使用 **STDIO（推荐选项 1）** 或 SSE（不推荐）作为传输方式。

```bash
# 选项 1：使用 MCP Inspector（推荐）
# 注意路径中不要出错
.\.venv\Scripts\fastmcp.exe dev oceanserver.py
# 或：fastmcp dev oceanserver.py
# 可在 http://127.0.0.1:6274 进行测试

# 选项 2：使用 uv run 启动
uv run --directory FULL_PATH/OceanGPT/mcp_server fastmcp run FULL_PATH/OceanGPT/mcp_server/oceanserver.py
# 注意 Win 和 Mac 的路径符号不同

# 选项 3：使用 python 启动 SSE 模式
# 修改 oceanserver.py 文件：
# 将 mcp.run() 替换为 mcp.run(transport="sse")
python oceanserver.py
# 默认本地 SSE URL 为 http://127.0.0.1:8000
```

1. **选择 MCP 客户端并添加 `json` 配置**

具体操作请参见下一章节（客户端调用）。

配置文件示例：

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



## 客户端调用

> 支持本地与远程服务器，本章则以 [Cursor](https://www.cursor.com/cn) 为例。

所有 MCP 客户端仅需修改其 `json` 配置文件即可。

**本地配置示例：**

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

其中 `FULL_PATH` 是本地路径。注意 Windows 和 Linux、MacOS 路径符号不同，请确保路径无误。

**远程配置示例：**

```json
{
  "mcpServers": {
    "OceanGPT_Remote": {
      "url": "OUR_REMOTE_URL"
    }
  }
}
```

👇 目前 `OUR_REMOTE_URL` 为：

```
https://oceangpt-mcp.onrender.com/sse
```

或者：

```
https://oceangpy-c-cite-mwlqnwjval.cn-hangzhou.fcapp.run/sse
```

（请选用更稳定的那个）
