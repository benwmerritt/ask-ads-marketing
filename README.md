# Ask Ads & Marketing - RAG MCP Server

Production-ready **Model Context Protocol (MCP)** server that exposes expert paid advertising and marketing strategy content through a searchable RAG (Retrieval-Augmented Generation) system.

## Overview

This server provides AI-powered access to paid ads, social media marketing, email strategy, and ad spend optimization content from expert marketing practitioners through the MCP protocol, enabling Claude and other LLM applications to retrieve and synthesize marketing insights.

## Features

- 🎯 **Marketing Expertise**: Access expert frameworks for Meta ads, email campaigns, and social strategy
- 🔍 **Smart Search**: Vector-based semantic search across marketing content transcripts
- 📚 **Citation-Based Answers**: All responses include source references
- 🔐 **Secure API Key Auth**: Production-ready authentication for remote access
- 🚀 **Railway Deployment**: One-click deploy with persistent ChromaDB storage
- ⚡ **Fast Retrieval**: Optimized chunking and embedding strategies

## Tools

### `ask_ads_marketing`
Ask questions about paid advertising, Meta ads, email marketing, social media strategy, and ad spend optimization.

**Parameters:**
- `question` (required): Your marketing/ads question
- `top_k` (optional, default: 18): Number of content chunks to retrieve
- `max_tokens` (optional, default: 2600): Maximum response length
- `user_context` (optional): Additional context to tailor recommendations
- `response_style` (optional): `'concise'`, `'detailed'`, or `'comprehensive'`

**Returns:** `{answer, sources, confidence}`

**Example Questions:**
- "How do I structure Meta ad campaigns for e-commerce?"
- "What's the best email sequence for SaaS onboarding?"
- "How do I improve my Instagram engagement rate?"
- "What metrics should I track for paid social ROI?"
- "How do I scale profitable ad campaigns?"
- "What's the optimal Facebook ad budget allocation strategy?"
- "How do I reduce my cost per acquisition on Meta ads?"

### `about`
Get information about this MCP server and its capabilities.

## Setup

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add transcripts:**
   Place marketing/ads transcript `.md` files in `/transcripts/` directory

3. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY=your_openai_key
   ```

4. **Run ingestion:**
   ```bash
   python scripts/ingest.py
   ```

5. **Start MCP server (stdio):**
   ```bash
   python mcp_server.py
   ```

### Railway Deployment

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Ask Ads & Marketing MCP"
   git remote add origin https://github.com/YOUR_USERNAME/ask-ads-marketing.git
   git push -u origin main
   ```

2. **Deploy on Railway:**
   - Create new project from GitHub repo
   - Add environment variables:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `VALID_API_KEYS`: Comma-separated API keys for MCP access
     - `MCP_SERVER_URL`: Your Railway app URL
     - `ENVIRONMENT`: `production`
   - (Optional) Add Railway volume at `/data` for persistent ChromaDB

3. **First deployment:**
   - Railway will build the Docker image
   - Ingestion runs automatically (~10-15 minutes for ~500 transcripts)
   - Server starts and exposes MCP endpoint

4. **Test the deployment:**
   ```bash
   curl https://your-app.railway.app/health
   ```

## Configuration

### `configs/ask-ads-marketing-deploy.yaml`

Main configuration file for production deployment:

- **knowledge_base**: Creator info and personality (data-driven, ROI-focused)
- **data_sources**: Transcript locations and preferences
- **embeddings**: OpenAI embeddings configuration
- **chunking**: Smart chunking for optimal retrieval
- **search**: Hybrid search with MMR diversity
- **generation**: GPT-4o-mini for answer synthesis
- **chroma**: Vector database settings

## Content Format

Transcripts should be Markdown files with optional YAML frontmatter:

```markdown
---
title: "Lesson Title"
channel: "@ChannelName"
url: https://youtube.com/watch?v=VIDEO_ID
date_processed: 2025-01-15
---

## Transcript
Full transcript text goes here...
```

## MCP Protocol

This server implements the [Model Context Protocol](https://modelcontextprotocol.io/) for seamless integration with Claude and other LLM applications.

**Remote Access URL:**
```
https://your-app.railway.app/mcp?apiKey=YOUR_API_KEY
```

## Architecture

- **FastMCP**: MCP protocol implementation
- **ChromaDB**: Vector database for embeddings
- **OpenAI**: Embeddings (text-embedding-3-small) + Generation (gpt-4o-mini)
- **FastAPI**: HTTP server for remote access
- **Railway**: Cloud deployment platform

## Topics Covered

- Paid advertising (Meta/Facebook, Instagram, TikTok, YouTube)
- Social media marketing and organic growth
- Email marketing campaigns and automation
- Ad spend optimization and budget allocation
- Conversion funnel design and CRO
- Landing page strategy and optimization
- Audience targeting and segmentation
- Creative testing and ad copy frameworks
- Analytics, attribution, and performance tracking
- Scaling profitable campaigns

## Performance

- **Retrieval speed**: < 1 second for most queries
- **Answer generation**: 2-5 seconds (varies by response_style)
- **Database size**: ~500+ marketing/ads transcripts
- **Uptime**: 99%+ on Railway with persistent storage

## License

Private - For authorized use only

## Author

Benjamin Merritt

## Support

For issues or questions, contact: [Your contact info]
