# Research Agent

A research automation system built on the SmolaGents framework, designed to collect, analyze, and synthesize information from various sources to support research tasks.

## Features

- Natural language research query processing
- Information search and retrieval from multiple sources
- Data analysis and synthesis
- Structured reporting with citations
- Customizable research workflows

## Architecture

The system is built with a modular architecture:

- **Research Management Module**: Coordinates the research process
- **Analytics Module**: Processes and analyzes collected data
- **Interface Module**: Integrates with external systems
- **Storage Module**: Maintains context and intermediate results

## Tools

- Web search via API services
- URL content fetching
- PDF document analysis
- Data analysis and visualization
- Database management for collected data

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your API keys
6. Run the application: `python main.py`

## Usage

```bash
# Basic usage
python main.py "Research query"

# With configuration file
python main.py --config config.json

# Export results to file
python main.py "Research query" --output report.md
```

## Development

- Follow the modular architecture
- Document code with docstrings
- Add tests for new components
- Keep API keys in environment variables

## License

MIT
