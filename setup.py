from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sql_synthesizer",
    version="0.2.2",
    author="SQL Synthesizer Team",
    author_email="team@sqlsynthesizer.com",
    description="Natural-language-to-SQL agent with automatic schema discovery and query validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/sql-synthesizer",
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "ruff>=0.0.270",
        ],
        "redis": ["redis>=4.0.0"],
        "memcached": ["pymemcache>=4.0.0"],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "query-agent=query_agent:main",
            "query-agent-web=sql_synthesizer.webapp:main",
            "sql-synthesizer-db=scripts.db_manager:main",
        ]
    },
    include_package_data=True,
    package_data={
        "sql_synthesizer": [
            "templates/*.html",
            "static/*",
            "database/migrations/*.sql",
        ],
    },
    zip_safe=False,
)
