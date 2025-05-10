#!/usr/bin/env python
"""
Test script for the research agent.
"""

import json
from agent.research_agent import ResearchAgent

def main():
    # Load the configuration
    with open('config/default_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Initialize the agent
    agent = ResearchAgent(config)
    
    # Process a Ukrainian query
    query = "досліди ринок смартфонів в Україні"
    print(f"Research Query: {query}")
    
    # Process the query
    results = agent.process_query(query)
    
    # Print the results
    print("\nResults:")
    print(f"Summary: {results.get('summary', 'No summary available')}")

if __name__ == "__main__":
    main()
