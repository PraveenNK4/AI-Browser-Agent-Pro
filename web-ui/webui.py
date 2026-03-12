from dotenv import load_dotenv
load_dotenv()
import argparse
from src.webui.interface import theme_map, create_ui
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


def main():
    parser = argparse.ArgumentParser(description="Gradio WebUI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    args = parser.parse_args()

    args = parser.parse_args()

    # Suppress security warnings as requested
    import logging
    
    # Global sensitive values for redaction
    SENSITIVE_VALUES = set()

    class RedactingFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # Suppress specific noisy warnings
            if any(s in msg for s in ["is not locked down", "insecure settings", "Failed to update state"]):
                return False
            
            # Redact sensitive values
            if SENSITIVE_VALUES:
                import re
                processed_msg = msg
                for val in SENSITIVE_VALUES:
                    if val and len(str(val)) > 2: # Only redact non-trivial strings
                        processed_msg = processed_msg.replace(str(val), "******")
                
                # Update the record message directly
                record.msg = processed_msg
                record.args = () # Clear args to prevent re-formatting
                
            return True

    # Register the filter globally
    redacting_filter = RedactingFilter()
    
    # Attach to all relevant loggers
    for logger_name in ["agent", "browser_use", "browser", "registry", "src.controller", "src.utils"]:
        logging.getLogger(logger_name).addFilter(redacting_filter)
    
    # Expose for other modules to update
    import builtins
    builtins.redacting_filter_values = SENSITIVE_VALUES

    demo = create_ui(theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    main()
