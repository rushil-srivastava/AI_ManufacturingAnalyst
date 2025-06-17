# FactoryTwin

A manufacturing analytics system that uses AI to generate and execute SQL queries for manufacturing data analysis.

## Features

- Natural language to SQL query generation
- Question classification (Descriptive, Judgement, Suggestion)
- Manufacturing-specific analytics
- Integration with MySQL database
- Support for complex manufacturing queries
- Built-in documentation and training data

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure database connection in `mistral_v2.py`
4. Run the application:
   ```bash
   python mistral_v2.py
   ```

## Requirements

- Python 3.8+
- MySQL database
- Required Python packages (see requirements.txt)

## Usage

1. Start the application
2. Enter your question in natural language
3. The system will:
   - Classify your question
   - Generate appropriate SQL
   - Execute the query
   - Provide analysis and insights

## License

MIT License 