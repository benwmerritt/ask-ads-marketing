#!/usr/bin/env python3
"""
Remote MCP Server for Ask Ads & Marketing

Production-ready remote MCP server that exposes paid advertising and marketing
strategy content to LLM applications like Claude Web/iOS via API key authentication.

Endpoints:
  - POST /mcp - MCP protocol endpoint (Streamable HTTP transport)
  - DELETE /mcp - Session cleanup
  - GET /health - Health check
  - GET / - Service info

Usage:
  VALID_API_KEYS=your_key uvicorn remote_mcp_server:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse

# Import the MCP server instance from local MCP server
# This already has all tools registered
import mcp_server

# Use the existing MCP server from mcp_server.py
# All tools are already registered there (ask_ads_marketing, about)
mcp = mcp_server.mcp

# Create MCP ASGI app (Streamable HTTP transport)
# IMPORTANT: when this app is mounted at "/mcp" below, the mounted path prefix is
# stripped before routing inside the FastMCP app. Therefore the FastMCP app must
# expose its transport at "/" so requests to "/mcp" resolve correctly.
mcp_app = mcp.http_app(path="/")

# Create FastAPI app with MCP lifespan (CRITICAL for MCP session initialization)
app = FastAPI(
    title="Ask Ads & Marketing Remote MCP",
    version="1.0.0",
    description="Remote Model Context Protocol server for paid advertising and marketing strategy content",
    lifespan=mcp_app.lifespan,  # Use MCP's lifespan for proper initialization
)

# Authentication middleware for MCP endpoints
class MCPAuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for MCP endpoints.
    Validates API key from query parameters.
    """
    async def dispatch(self, request: Request, call_next):
        # Only authenticate /mcp/* paths
        if request.url.path.startswith("/mcp"):
            # Check for API key in query parameters
            api_key = request.query_params.get("apiKey")

            if api_key:
                # Validate API key
                valid_keys = os.getenv("VALID_API_KEYS", "").split(",")
                valid_keys = [k.strip() for k in valid_keys if k.strip()]

                if not valid_keys:
                    return StarletteJSONResponse(
                        {"error": "API key authentication not configured"},
                        status_code=500
                    )

                if api_key not in valid_keys:
                    return StarletteJSONResponse(
                        {"error": "Invalid API key"},
                        status_code=401
                    )
            else:
                # API key mode - require API key
                return StarletteJSONResponse(
                    {"error": "API key required. Add ?apiKey=YOUR_KEY to the URL"},
                    status_code=401
                )

        response = await call_next(request)
        return response

# Add authentication middleware
app.add_middleware(MCPAuthMiddleware)

# CORS middleware for remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id", "WWW-Authenticate", "Content-Type"],
)

# Mount the MCP ASGI app at /mcp endpoint
app.mount("/mcp", mcp_app)


class MCPPathRouter:
    """ASGI wrapper that makes the mounted MCP app respond at /mcp without redirects."""

    def __init__(self, main_app, mounted_app, mount_path: str = "/mcp") -> None:
        self.main_app = main_app
        self.mounted_app = mounted_app
        self.mount_path = mount_path

    def __getattr__(self, item):
        return getattr(self.main_app, item)

    async def __call__(self, scope, receive, send):  # type: ignore[override]
        if scope.get("type") == "http" and scope.get("path") == self.mount_path:
            # Rewrite the scope as if the MCP app were mounted at "/"
            new_scope = dict(scope)
            new_scope["path"] = "/"
            new_scope["raw_path"] = b"/"
            new_scope["root_path"] = scope.get("root_path", "") + self.mount_path
            await self.mounted_app(new_scope, receive, send)
            return
        await self.main_app(scope, receive, send)


# ==================== Health & Info Endpoints ====================


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "ok": True,
        "service": "Ask Ads & Marketing Remote MCP",
        "version": "1.0.0",
        "mcp_protocol_version": "2025-03-26",
        "tools_count": len(mcp._tool_manager._tools),
        "status": "healthy"
    }


@app.get("/")
async def root():
    """Root endpoint with service info"""
    base_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

    return {
        "name": "Ask Ads & Marketing Remote MCP",
        "version": "1.0.0",
        "description": "Remote MCP server for paid advertising and marketing strategy content",
        "topic": "Paid advertising, social media marketing, ad spend optimization, and email strategy",
        "endpoints": {
            "mcp": f"{base_url}/mcp",
            "health": f"{base_url}/health",
        },
        "tools": ["ask_ads_marketing", "about"],
        "authentication": "API key required in query parameter: ?apiKey=YOUR_KEY",
        "documentation": f"{base_url}/docs",
    }


fastapi_app = app  # keep reference to the FastAPI instance for testing/inspection
app = MCPPathRouter(app, mcp_app)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "remote_mcp_server:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
    )
