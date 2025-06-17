```markdown
**JobApplicationBot**

**Overview**

This repository contains an automated job application agent that uses OpenAI's API and Playwright for end-to-end interaction with job portals.

**Prerequisites**

- Python 3.8 or higher  
- Git

**Setup Instructions**

Follow these steps to get the project up and running locally:

1. **Create and activate a Python virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

2. **Export your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

3. **Install project dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Playwright browsers**
   ```bash
   playwright install
   ```

5. **Run the auto-generation script**
   ```bash
   python autogen.py
   ```

## Usage

After setup, the `autogen.py` script will launch and begin automating job applications based on your configuration. Logs will appear in the console.

## Configuration

- Ensure your `.env` file or environment variables include `OPENAI_API_KEY`.  
- Modify any settings in `config.json` (if applicable) before running.
