# Agent System - Business Idea Validator

## Project Overview

This is an advanced multi-agent system designed for business idea validation. The system leverages AI and web scraping to analyze market demand, user pain points, and competitive landscape by collecting and analyzing data from social media platforms, particularly Xiaohongshu (Little Red Book/Red).

### Architecture

The system follows a modular architecture with the following key components:

1. **Agents**: Core intelligence units that perform specific tasks
2. **MCP Servers**: Microservice-based servers for different capabilities
3. **Context Store**: Shared state management between agents
4. **Models**: Data structures and business logic models
5. **Config**: Configuration management system

### Core Components

#### 1. Base Agent (`agents/base_agent.py`)
- Abstract base class for all agents
- Provides unified interfaces for MCP calls, LLM interactions, and progress tracking
- Implements lifecycle management (start, stop, pause, resume)
- Includes metrics tracking and checkpoint management

#### 2. Context Store (`agents/context_store.py`)
- Centralized storage for shared context between agents
- Manages run contexts, progress tracking, and agent states
- Supports both in-memory and file-based persistence
- Implements TTL-based cleanup for expired data

#### 3. Configuration Manager (`agents/config.py`)
- Handles configuration loading from YAML, JSON, and environment variables
- Supports different configuration types (XHS, LLM, Storage, Orchestrator)
- Provides default values and environment variable overrides

#### 4. MCP Servers (`mcp_servers/`)
- **XHS Server**: Interfaces with TikHub API for Xiaohongshu data
- **LLM Server**: Provides LLM capabilities for analysis and generation
- **Storage Server**: Manages checkpoint persistence and data storage

#### 5. Models (`models/`)
- **Agent Models**: Task results, progress updates, execution plans
- **Context Models**: Run contexts, agent states
- **Business Models**: Xiaohongshu notes, comments, analysis results

## Building and Running

### Prerequisites
- Python 3.8+
- TikHub API token for Xiaohongshu data access
- LLM API key (OpenAI, etc.)

### Setup
1. Install required dependencies (not explicitly listed but likely includes `requests`, `pydantic`, `yaml`, `asyncio`)
2. Set up environment variables:
   - `TIKHUB_TOKEN`: Your TikHub API token
   - `OPENAI_API_KEY`: Your OpenAI API key (or equivalent)
   - `OPENAI_BASE_URL`: API endpoint URL

### Running Tests
Execute the integration tests to verify system functionality:
```bash
python tests/test_integration.py
```

### Main Execution Flow
The orchestrator agent manages the complete business validation workflow:
1. **Keyword Generation**: Generate relevant search keywords
2. **Data Scraping**: Collect Xiaohongshu posts and comments
3. **Data Analysis**: Analyze content for market signals and user pain points
4. **Report Generation**: Create comprehensive validation report

## Development Conventions

### Code Structure
- All agents inherit from `BaseAgent`
- MCP servers follow a consistent interface pattern with `call_tool` method
- Models use Pydantic for data validation
- Async/await patterns for I/O operations
- Thread-safe context store operations

### Error Handling
- Comprehensive exception handling with logging
- Retry mechanisms with exponential backoff
- Graceful degradation when services are unavailable

### Configuration
- Centralized configuration management
- Environment variable overrides
- YAML/JSON configuration file support
- Default values for all settings

### Testing
- Integration tests verify end-to-end functionality
- Tests cover MCP servers, context store, and configuration
- Mock-friendly architecture for unit testing

## Key Features

### 1. Multi-Agent Architecture
- Orchestrator coordinates the validation workflow
- Specialized agents for different tasks (keyword generation, scraping, analysis, reporting)
- Agent delegation and communication mechanisms

### 2. MCP (Microservice Control Protocol)
- Standardized interface for different service types
- Pluggable architecture for adding new capabilities
- Centralized service management

### 3. Progress Tracking
- Real-time progress updates during execution
- Detailed progress history and metrics
- Callback mechanisms for external progress monitoring

### 4. Checkpoint Management
- Save/restore functionality for long-running processes
- Automatic checkpointing at defined intervals
- Resume from failure capabilities

### 5. Data Models
- Rich data models for Xiaohongshu content
- Comprehensive analysis results structure
- Validation result aggregation

## Available Skills

The system includes a comprehensive set of skills organized by functional areas:

### Keyword Skills
- `generate_keywords_skill`: Generate search keywords based on business idea
- `refine_keywords_skill`: Optimize existing keywords based on feedback
- `validate_keywords_skill`: Validate keyword quality and relevance

### Scraper Skills
- `search_posts_skill`: Search Xiaohongshu posts by keyword
- `get_comments_skill`: Get comments for a specific post
- `batch_get_comments_skill`: Get comments for multiple posts in batch
- `batch_scrape_skill`: Perform batch scraping of posts and comments

### Analyzer Skills
- `analyze_post_skill`: Analyze individual posts for relevance and insights
- `analyze_comments_skill`: Analyze comments for user sentiment and themes
- `batch_analyze_posts_skill`: Batch analysis of multiple posts
- `generate_combined_analysis_skill`: Generate comprehensive market analysis

### Reporter Skills
- `generate_text_report_skill`: Create text format validation reports
- `generate_html_report_skill`: Create HTML format validation reports with styling
- `save_report_skill`: Save reports to file system

## File Structure
```
agent_system/
├── __init__.py
├── .env
├── README.md
├── agents/                 # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py       # Base agent class
│   ├── config.py           # Configuration management
│   ├── context_store.py    # Shared context management
│   ├── orchestrator.py     # Main orchestrator agent
│   ├── subagents/          # Specialized agents
│   └── skills/             # Agent capabilities
├── config/                 # Configuration files
├── docs/                   # Documentation
├── mcp_servers/            # MCP server implementations
│   ├── __init__.py
│   ├── xhs_server.py       # Xiaohongshu server
│   ├── llm_server.py       # LLM server
│   └── storage_server.py   # Storage server
├── models/                 # Data models
│   ├── __init__.py
│   ├── agent_models.py     # Agent-related models
│   ├── context_models.py   # Context models
│   └── business_models.py  # Business domain models
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_integration.py # Integration tests
├── utils/                  # Utility functions
└── agent_context/          # Runtime context storage
    └── checkpoints/        # Checkpoint files
```

## Environment Variables
- `TIKHUB_TOKEN`: API token for TikHub Xiaohongshu service
- `OPENAI_API_KEY`: API key for LLM service
- `OPENAI_BASE_URL`: Base URL for LLM API (default: OpenAI)
- `REDIS_URL`: Redis connection string (if using Redis storage)

## API Integration
The system integrates with:
- TikHub API for Xiaohongshu data access
- LLM APIs (OpenAI, etc.) for analysis and generation
- File system or Redis for persistent storage

## Development Notes
- The system is currently in Phase 1 (Basic Infrastructure) as per the development plan
- Future phases include implementation of subagents and orchestrator
- The system is designed to be extensible for additional social media platforms
- Error handling and retry mechanisms are built into the base agent class