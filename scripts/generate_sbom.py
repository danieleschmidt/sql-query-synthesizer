#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generator for SQL Synthesizer
Generates SPDX-compliant SBOM documents for security and compliance
"""

import json
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from pkg_resources import get_distribution
import uuid

def get_git_commit():
    """Get current git commit hash"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"

def get_package_info():
    """Extract installed package information"""
    packages = []
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'], 
                              capture_output=True, text=True, check=True)
        pip_packages = json.loads(result.stdout)
        
        for pkg in pip_packages:
            packages.append({
                "name": pkg["name"],
                "version": pkg["version"],
                "type": "python-package"
            })
    except Exception as e:
        print(f"Error getting package info: {e}")
    
    return packages

def calculate_file_hashes(project_root):
    """Calculate SHA256 hashes for key project files"""
    files_to_hash = [
        "pyproject.toml",
        "requirements.txt", 
        "requirements-dev.txt",
        "Dockerfile",
        "docker-compose.yml"
    ]
    
    file_hashes = {}
    for file_path in files_to_hash:
        full_path = Path(project_root) / file_path
        if full_path.exists():
            with open(full_path, 'rb') as f:
                content = f.read()
                sha256_hash = hashlib.sha256(content).hexdigest()
                file_hashes[file_path] = sha256_hash
    
    return file_hashes

def generate_sbom():
    """Generate SPDX-compliant SBOM"""
    project_root = Path(__file__).parent.parent
    
    # Get project metadata
    try:
        dist = get_distribution('sql_synthesizer')
        project_version = dist.version
    except:
        project_version = "0.2.2"  # fallback
    
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "SQL-Query-Synthesizer-SBOM",
        "documentNamespace": f"https://github.com/yourorg/sql-synthesizer/sbom/{uuid.uuid4()}",
        "creationInfo": {
            "created": datetime.utcnow().isoformat() + "Z",
            "creators": ["Tool: sql-synthesizer-sbom-generator-1.0.0"],
            "licenseListVersion": "3.21"
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Package-sql-synthesizer",
                "name": "sql-synthesizer",
                "downloadLocation": "https://github.com/yourorg/sql-synthesizer",
                "filesAnalyzed": True,
                "versionInfo": project_version,
                "supplier": "Organization: SQL Synthesizer Team",
                "licenseConcluded": "MIT",
                "licenseDeclared": "MIT",
                "copyrightText": "Copyright (c) 2025 SQL Synthesizer Team",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/sql-synthesizer@{project_version}"
                    }
                ]
            }
        ],
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Package-sql-synthesizer"
            }
        ]
    }
    
    # Add dependency packages
    packages = get_package_info()
    for i, pkg in enumerate(packages):
        pkg_id = f"SPDXRef-Package-{pkg['name'].replace('-', '_')}"
        sbom["packages"].append({
            "SPDXID": pkg_id,
            "name": pkg["name"],
            "versionInfo": pkg["version"],
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{pkg['name']}@{pkg['version']}"
                }
            ]
        })
        
        # Add dependency relationship
        sbom["relationships"].append({
            "spdxElementId": "SPDXRef-Package-sql-synthesizer",
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": pkg_id
        })
    
    # Add file information
    file_hashes = calculate_file_hashes(project_root)
    git_commit = get_git_commit()
    
    sbom["annotations"] = [
        {
            "annotationType": "OTHER",
            "annotator": "Tool: sql-synthesizer-sbom-generator",
            "annotationDate": datetime.utcnow().isoformat() + "Z",
            "annotationComment": f"Generated from git commit: {git_commit}"
        }
    ]
    
    # Add file checksums as annotations
    for file_path, file_hash in file_hashes.items():
        sbom["annotations"].append({
            "annotationType": "OTHER",
            "annotator": "Tool: sql-synthesizer-sbom-generator",
            "annotationDate": datetime.utcnow().isoformat() + "Z",
            "annotationComment": f"SHA256 checksum for {file_path}: {file_hash}"
        })
    
    return sbom

def main():
    """Main function to generate and save SBOM"""
    try:
        sbom = generate_sbom()
        
        # Save SBOM to file
        output_file = Path(__file__).parent.parent / "sbom.json"
        with open(output_file, 'w') as f:
            json.dump(sbom, f, indent=2)
        
        print(f"SBOM generated successfully: {output_file}")
        print(f"Total packages documented: {len(sbom['packages'])}")
        
        # Also generate a human-readable summary
        summary_file = Path(__file__).parent.parent / "sbom-summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Software Bill of Materials Summary\n")
            f.write("===================================\n\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
            f.write(f"Project: sql-synthesizer\n")
            f.write(f"Total Dependencies: {len(sbom['packages']) - 1}\n\n")
            
            f.write("Main Dependencies:\n")
            for pkg in sbom['packages'][1:11]:  # Show first 10 deps
                f.write(f"  - {pkg['name']} ({pkg['versionInfo']})\n")
            
            if len(sbom['packages']) > 11:
                f.write(f"  ... and {len(sbom['packages']) - 11} more\n")
        
        print(f"Human-readable summary: {summary_file}")
        
    except Exception as e:
        print(f"Error generating SBOM: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()