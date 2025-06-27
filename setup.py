from setuptools import setup, find_packages

setup(
    name="sql_synthesizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["SQLAlchemy>=2.0", "PyYAML", "openai"],
    entry_points={
        "console_scripts": [
            "query-agent=query_agent:main",
        ]
    },
)
