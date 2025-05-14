# DemoAgents - AI Agents with LangChain and LangGraph

This repository contains example projects demonstrating how to build AI agents and workflows using LangChain and LangGraph with Azure OpenAI. These examples showcase practical applications of AI agents for document summarization and travel planning.

## Project Overview

This project contains these main implementations:

1. **Summarize LangChain**: A simple script demonstrating how to use LangChain to summarize text documents using Azure OpenAI.

2. **Travel Agent Backend**: A complex agent based on LangGraph that plans travel itineraries using multiple specialized sub-agents.

3. **Travel Agent Web UI**: A Streamlit-based web interface that provides real-time visual feedback during the travel planning process.

## Requirements

- Python 3.9+
- An Azure account with Azure OpenAI service configured
- Streamlit (for the web interface)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/erinhoeam/demo-agent.git
cd demo-agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure a `.env` file in the project root with your Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2023-07-01-preview
```

## Usage

### Document Summarizer

The `summarize_langchain.py` script allows you to summarize text documents using Azure OpenAI.

```bash
python summarize_langchain.py files/azure_ai_services.txt --output summary.txt
```

Options:

- `file_path`: Path to the file or directory containing text files to summarize
- `--output` or `-o`: Path to save the summary output
- `--verbose` or `-v`: Enable detailed logging

### Travel Planning Agent

#### Command Line Interface

The `travel_agent.py` script implements a travel planning agent with LangGraph that coordinates specialized sub-agents.

```bash
python travel_agent.py "Paris, France" --output paris_travel_plan.json
```

#### Web Interface (Streamlit)

The `chat.py` script provides an interactive web interface for the travel planning agent using Streamlit.

```bash
streamlit run chat.py
```

This interface features:

- Real-time streaming updates as the agent works
- Progress bars and visual feedback for each stage
- Typewriter effects for an engaging user experience
- Chat history tracking
- Configurable display speeds (fast, normal, slow)
- Mobile-friendly responsive design

Options:

- `destination`: Travel destination to plan for
- `--output` or `-o`: Path to save the travel plan as JSON
- `--verbose` or `-v`: Enable detailed logging
- `--debug` or `-d`: Include debug information in the output
- `--max-steps`: Maximum number of steps in the agent workflow (default: 10)

## Architecture

### Document Summarizer

The `summarize_langchain.py` script uses:

- `TextLoader` and `DirectoryLoader` to load documents
- `ChatPromptTemplate` for prompt formatting
- `AzureChatOpenAI` for response generation
- Retry logic with `tenacity` for robustness

### Travel Planning Agent

#### Backend Architecture

The `travel_agent.py` implements a multi-agent architecture using LangGraph:

1. **Router Agent**: Coordinates the workflow by deciding which sub-agent to call
2. **Attractions Agent**: Identifies tourist attractions for the destination
3. **Route Planner Agent**: Creates an efficient route to visit the attractions
4. **Finalizer Agent**: Consolidates all information into a complete travel plan

The workflow follows a sequential execution between specialized agents, with each agent updating the shared state. The implementation has been optimized to handle state key conflicts and ensure proper validation of agent outputs.

#### Frontend Architecture

The `chat.py` implements a Streamlit-based web interface that:

1. Presents a chat-like interface for user input
2. Provides real-time updates on the planning process
3. Manages application state with Streamlit session state
4. Implements visual effects like typewriter text and loading animations
5. Displays a progress bar indicating the overall completion status
6. Formats the final travel plan with markdown, emojis, and clear structure

## Key Features

- API key authentication for Azure OpenAI
- Retry logic with exponential backoff
- Robust error handling
- Validation of intermediate results
- Detailed logging
- Debug mode for troubleshooting
- Streamlit-based interactive UI
- Real-time streaming updates
- Visual progress indicators
- Typewriter text effects

## Project Structure

```
DemoAgents/
│
├── summarize_langchain.py     # Document summarization script
├── travel_agent.py            # Travel agent implementation with LangGraph
├── chat.py                    # Streamlit web interface for travel agent
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── .env                       # Configuration file (not versioned)
│
├── .vscode/                   # VS Code configuration
│   └── launch.json            # Debug configurations
│
├── files/
│   └── azure_ai_services.txt  # Sample document for summarization
│
└── paris_travel_plan.json     # Example output from travel agent
```

## Output Examples

### Document Summary

The generated summary includes the main points of the input document, presented concisely.

### Travel Plan

The generated travel plan contains:

- Destination overview
- Day-by-day itinerary
- Practical tips for the traveler
- Cost estimates
- Packing suggestions

## Implementation Notes

The travel agent uses a sequential execution approach instead of a fully graph-based workflow to avoid state conflicts. This implementation ensures proper validation of agent outputs and includes retry logic for more reliable results.

### Backend Improvements

Key improvements in the backend implementation include:

- Renamed node from "attractions" to "attractions_agent" to avoid state key conflicts
- Enhanced route planner with numbered attractions and explicit formatting
- Added validation of agent outputs to prevent empty or malformed responses
- Improved error handling and debug output

### Streamlit Interface Features

The Streamlit interface adds significant user experience improvements:

- **Real-time Feedback**: Visual indicators show the current stage of travel planning
- **Streaming Text Effect**: Typewriter-style text generation creates an engaging experience
- **Progress Tracking**: Progress bars indicate completion percentage for each step
- **Custom Formatting**: Attractions and itineraries are formatted with emojis and clear structure
- **User Preferences**: Configurable display speed (fast/normal/slow) in the sidebar
- **Error Handling**: User-friendly error messages with detailed technical information
- **Responsive Design**: Works well on both desktop and mobile devices

## License

MIT

## Contributing

Contributions are welcome! Please submit pull requests or open issues to discuss improvements.
