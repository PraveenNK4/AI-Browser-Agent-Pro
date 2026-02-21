"""CLI interface for process orchestration.

Usage:
    python -m src.orchestrator.cli "Mandatory Process: ...\\n\\nProcess 1: ..."
    python -m src.orchestrator.cli --file processes.txt
"""
import asyncio
import argparse
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main CLI entry point for orchestration."""
    parser = argparse.ArgumentParser(
        description="Process orchestration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct prompt
  python -m src.orchestrator.cli "Mandatory Process: Login\\n- Step 1"
  
  # From file
  python -m src.orchestrator.cli --file processes.txt
  
  # With output directory
  python -m src.orchestrator.cli "Process 1: Task" --output ./results
"""
    )
    
    parser.add_argument('prompt', nargs='?', help='Process description (or use --file)')
    parser.add_argument('--file', help='Read processes from file')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--max-steps', type=int, default=25, help='Maximum steps per process')
    parser.add_argument('--model', default='qwen2.5:7b', help='LLM model name')
    parser.add_argument('--base-url', help='LLM base URL')
    
    args = parser.parse_args()
    
    # Get prompt
    if args.file:
        try:
            prompt = Path(args.file).read_text()
            logger.info(f"Loaded processes from: {args.file}")
        except FileNotFoundError:
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.print_help()
        sys.exit(1)
    
    try:
        from src.orchestrator import parse_processes, run_all_processes, generate_process_summary
        from src.orchestrator.report_generator import save_summary_report, generate_combined_playwright_script
        from src.browser.custom_browser import CustomBrowser
        from src.utils import llm_provider
        from browser_use.browser.browser import BrowserConfig
        from browser_use.browser.context import BrowserContextConfig
        
        # Parse processes
        processes = parse_processes(prompt)
        logger.info(f"Parsed {len(processes)} processes")
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {args.model}")
        llm = llm_provider.get_llm_model(
            provider="ollama",
            model_name=args.model,
            temperature=0.3,
            base_url=args.base_url,
        )
        
        # Create browser
        logger.info("Creating browser")
        browser = CustomBrowser(config=BrowserConfig(headless=False))
        
        # Run orchestration
        logger.info(f"Starting orchestration with {len(processes)} processes")
        results, output_dir = await run_all_processes(
            processes=processes,
            browser=browser,
            llm=llm,
            orchestration_output_dir=args.output,
            browser_context_config=BrowserContextConfig(),
            max_steps=args.max_steps,
        )
        
        # Generate reports
        summary = generate_process_summary(results)
        save_summary_report(summary, output_dir)
        generate_combined_playwright_script(results, f"{output_dir}/combined_script.py")
        
        # Print summary
        print(f"\n{'='*80}")
        print("ORCHESTRATION COMPLETE")
        print(f"{'='*80}")
        print(summary)
        print(f"\nOutput: {output_dir}")
        
        # Cleanup
        await browser.close(force=True)
        
        # Return status
        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count
        logger.info(f"Results: {success_count} succeeded, {failed_count} failed")
        sys.exit(0 if failed_count == 0 else 1)
        
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
