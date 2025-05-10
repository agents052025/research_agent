#!/usr/bin/env python3
"""
Research Agent - An AI-powered research assistant based on SmolaGents framework.
This is the main entry point for the application.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from dotenv import load_dotenv

# Load project modules
from agent.research_agent import ResearchAgent
from config.config_manager import ConfigManager
from core.logging_manager import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Research Agent - An AI-powered research assistant"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query to process"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for research results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    return parser.parse_args()


def setup_environment():
    """Set up the application environment."""
    # Load environment variables
    load_dotenv()
    
    # Check for required API keys
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    
    if missing_keys:
        console = Console()
        console.print(Panel(
            f"[bold red]Missing required API keys: {', '.join(missing_keys)}[/bold red]\n"
            f"Please add them to your .env file.", 
            title="Configuration Error"
        ))
        sys.exit(1)


def interactive_mode(config_manager):
    """Run the agent in interactive mode."""
    console = Console()
    console.print(Panel(
        "[bold green]Research Agent Interactive Mode[/bold green]\n"
        "Type your research query or 'exit' to quit.",
        title="Welcome"
    ))
    
    while True:
        query = console.input("[bold blue]Research Query:[/bold blue] ")
        if query.lower() in ("exit", "quit", "q"):
            break
            
        if not query.strip():
            continue
            
        try:
            # Initialize agent with current configuration
            agent = ResearchAgent(config_manager.get_config())
            
            # Process the query
            with Progress() as progress:
                task = progress.add_task("[green]Researching...", total=100)
                results = agent.process_query(query, progress_callback=lambda p: progress.update(task, completed=p))
                
            # Display results
            console.print(Panel(results["summary"], title="Research Summary"))
            
            # Ask if the user wants to save the results
            save = console.input("[bold yellow]Save results to file? (y/n):[/bold yellow] ")
            if save.lower() in ("y", "yes"):
                filename = console.input("[bold yellow]Enter filename:[/bold yellow] ") or "research_results.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(results["full_report"])
                console.print(f"[bold green]Results saved to {filename}[/bold green]")
                
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up environment
    setup_environment()
    
    # Set up logging
    logger = setup_logging(verbose=args.verbose)
    
    # Load configuration
    config_path = args.config or "config/default_config.json"
    config_manager = ConfigManager(config_path)
    
    try:
        # Interactive mode
        if args.interactive:
            interactive_mode(config_manager)
            return
            
        # Check if query is provided
        if not args.query:
            console = Console()
            console.print(
                "[bold yellow]No research query provided. "
                "Use --interactive mode or provide a query.[/bold yellow]"
            )
            sys.exit(1)
            
        # Initialize the research agent
        agent = ResearchAgent(config_manager.get_config())
        
        # Process the query
        console = Console()
        with console.status("[bold green]Researching...[/bold green]", spinner="dots"):
            results = agent.process_query(args.query)
        
        # Display results
        console.print(Panel(results["summary"], title="Research Summary"))
        
        # Save results if output file is specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(results["full_report"])
            
            console.print(f"[bold green]Full research report saved to {args.output}[/bold green]")
            
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}", exc_info=True)
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
