# Contributing to AgentOS

## Welcome

Thanks for your interest in contributing to AgentOS.
AgentOS is an open-source project licensed under Apache 2.0, and contributors of all skill levels are welcome.
Whether you are fixing a typo, improving tests, or building a major feature, your help is appreciated.

## Development Setup

Follow these steps to get a local development environment running:

1. Clone the repository:

   ```bash
   git clone https://github.com/<org>/agentos.git
   cd agentos
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

   - Windows (PowerShell):

     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

4. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

5. Copy environment variables template and add API keys:

   ```bash
   cp .env.example .env
   ```

6. Run the test suite:

   ```bash
   pytest
   ```

7. Run the web platform:

   ```bash
   python examples/run_web_builder.py
   ```

## Project Structure

The main Python package lives in `src/agentos/`.
Key modules include:

- `core` - foundational agent abstractions and runtime behavior
- `providers` - model/provider integrations and routing
- `governance` - safety, policy, budget, and guardrail logic
- `sandbox` - evaluation and scenario testing framework
- `rag` - retrieval-augmented generation components
- `scheduler` - scheduled jobs and orchestration
- `events` - event bus and trigger mechanisms
- `plugins` - extension interfaces and plugin management
- `auth` - authentication, authorization, and org usage
- `workflows` - multi-step workflow definitions and runners
- `marketplace` - agent/template registry and manifests
- `embed` - embedded SDK and widget support
- `templates` - prebuilt assistant and domain templates
- `web` - web app integration and serving layer

## How to Contribute

1. Check open issues, especially those labeled `good first issue`.
2. Fork the repository and create a feature branch:

   ```bash
   git checkout -b feat/your-change
   ```

3. Follow conventional commit messages:

   - `feat(module): description`
   - `fix(module): description`
   - `test(module): description`
   - `docs: description`

4. Write or update tests for any new behavior.
5. Run quality checks before submitting:

   ```bash
   pytest
   ruff check .
   ```

6. Open a pull request with a clear description of:
   - what changed
   - why it changed
   - how it was tested

## Good First Issues

If you are new to the project, start by visiting the GitHub Issues tab and filtering for `good first issue`.
Great places to begin include:

- adding or improving tests
- improving docstrings and inline documentation
- adding new tool or API usage examples
- polishing and expanding developer documentation

## Code Style

Please follow these code standards:

- Python 3.11+ compatibility is required
- type hints are required for new and updated code
- linting is enforced with `ruff`
- docstrings are expected on all public functions

Keeping style consistent makes code easier to review and maintain.

## Need Help?

Questions are welcome.
Please use GitHub Discussions for architecture questions, contributor guidance, and general support.
