from setuptools import setup, find_packages

setup(
    name="sql_synthesizer",
    version="0.2.2",
    packages=find_packages(),
    install_requires=["SQLAlchemy>=2.0", "PyYAML", "openai", "Flask", "prometheus-client", "python-dotenv", "sqlparse"],
    entry_points={
        "console_scripts": [
            "query-agent=query_agent:main",
            "query-agent-web=sql_synthesizer.webapp:main",
        ]
    },
)
