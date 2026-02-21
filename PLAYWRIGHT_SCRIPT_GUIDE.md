# Playwright Script Generation & Execution Guide

## Overview

When you run an automation task in the AI Browser Agent, the system automatically generates a standalone Playwright script that:

1. ✅ Executes all automation tasks
2. ✅ Captures screenshots at each step
3. ✅ Generates a professional DOCX report with screenshots

## Generated Files

After running a task, you get:

```
{run_id}/
├── {run_id}.json                        # Enriched history with DOM context
├── {run_id}.py                          # Standalone Playwright script
├── {run_id}.gif                         # Browser recording
├── {run_id}_report.docx                 # Agent execution report
│
└── {run_id}_screenshots/                # (Created when script runs)
    ├── 0.png
    ├── 1.png
    ├── 2.png
    └── ...

└── {run_id}_execution_log.json          # (Created when script runs)

└── {run_id}_playwright_report.docx      # (Created when script runs)
```

## Using the Generated Script

### Option 1: Run in Same Directory

```bash
cd tmp/agent_history/{run_id}
python {run_id}.py
```

### Option 2: Run with Credentials

```bash
set OTCS_USERNAME=myusername
set OTCS_PASSWORD=mypassword
python tmp/agent_history/{run_id}/{run_id}.py
```

### Option 3: Run from Anywhere

```bash
cd /any/directory
python /path/to/tmp/agent_history/{run_id}/{run_id}.py
```

## Script Features

### Built-in Report Generation

The generated script includes a complete report generator that:

- Creates a professional DOCX document
- Adds a summary section with execution status
- Creates a section for each automation step
- Embeds screenshots from each step
- Saves to `{run_id}_playwright_report.docx`

### Screenshot Capture

- Captures a screenshot after each action
- Saves screenshots as PNG files (0.png, 1.png, etc.)
- Automatically includes in the generated report

### Execution Logging

- Creates a JSON log with step-by-step execution details
- Useful for debugging and analysis
- Saved as `{run_id}_execution_log.json`

## Example Report Contents

When you run the script, the generated report includes:

**Execution Summary**

- Script name
- Execution status
- Date/time information

**Automation Steps** (one per step)

- Step number and action type
- Parameters used
- Screenshot from that step (if successful)

Example structure:

```
Step 1: go_to_url
[Screenshot of loaded page]

Step 2: input_text
[Screenshot showing filled text field]

Step 3: click_element_by_index
[Screenshot showing result of click]

...and so on
```

## Script Dependencies

The generated script only requires:

- Python 3.8+
- Playwright (for browser automation)
- python-docx (for report generation)

Install with:

```bash
pip install playwright python-docx
playwright install chromium
```

## Environment Variables

If your script uses credentials, set them before running:

```bash
# Windows
set OTCS_USERNAME=value
set OTCS_PASSWORD=value
python script.py

# Linux/Mac
export OTCS_USERNAME=value
export OTCS_PASSWORD=value
python script.py
```

## Output Files

### {run_id}\_screenshots/

Contains PNG images captured at each step. Numbered sequentially starting from 0.

### {run_id}\_execution_log.json

JSON file tracking execution flow:

```json
[
  {"step": 1, "action": "go_to_url"},
  {"step": 2, "action": "input_text"},
  {"step": 3, "action": "input_text"},
  ...
]
```

### {run_id}\_playwright_report.docx

Professional report document with:

- Execution summary
- Step-by-step details
- Embedded screenshots
- Formatted tables

## Troubleshooting

### Script fails to run

- Ensure Playwright is installed: `pip install playwright`
- Ensure browser is installed: `playwright install`
- Check environment variables are set if script uses them

### Report not generated

- Check that python-docx is installed: `pip install python-docx`
- Verify permissions to write files in the directory
- Check logs for specific error messages

### Screenshots not appearing in report

- Ensure screenshots directory was created
- Verify PNG files were saved correctly
- Check that python-docx can read the image files

## Advanced Usage

### Custom Report Location

The report is automatically saved alongside the screenshots directory.
To move it:

```bash
mv {run_id}_playwright_report.docx /desired/location/
```

### Rerunning with Different Credentials

Simply set new environment variables and rerun:

```bash
set OTCS_USERNAME=newuser
set OTCS_PASSWORD=newpass
python {run_id}.py
```

This will:

- Execute the same automation flow
- Capture new screenshots
- Generate a new report with new screenshots
