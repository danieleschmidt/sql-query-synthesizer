# SLSA Compliance Framework

Supply-chain Levels for Software Artifacts (SLSA) compliance documentation for SQL Query Synthesizer.

## Current SLSA Level Assessment

### SLSA Level 2 - Achieved ✅

**Build Requirements:**
- ✅ Version controlled source code (Git)
- ✅ Build service with defined service levels 
- ✅ Authenticated and service-generated provenance
- ✅ Immutable build environment

**Provenance Requirements:**
- ✅ Authenticated build steps
- ✅ Service-generated metadata
- ✅ Non-falsifiable provenance format

### SLSA Level 3 - Target State 🔄

**Source Requirements:**
- ✅ Version controlled source (GitHub)
- 🔄 Two-person review process (implemented via CODEOWNERS)
- ✅ Retained indefinitely (GitHub)

**Build Requirements:**
- 🔄 Hardened build platform (GitHub Actions + security hardening)
- 🔄 Non-falsifiable provenance format
- 🔄 Isolated build environment

## Implementation Strategy

### 1. Provenance Generation

#### GitHub Actions Integration
```yaml
# .github/workflows/slsa-provenance.yml
name: SLSA Provenance
on:
  push:
    tags: ['v*']
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.build.outputs.digest }}
    steps:
    - uses: actions/checkout@v4
    
    - name: Build package
      id: build
      run: |
        python -m build
        echo "digest=$(sha256sum dist/*.whl | cut -d' ' -f1)" >> $GITHUB_OUTPUT
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build-artifacts
        path: dist/

  provenance:
    needs: [build]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
    with:
      base64-subjects: "${{ needs.build.outputs.digest }}"
      upload-assets: true
```

### 2. Attestation Framework

#### Build Attestation
```json
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "subject": [{
    "name": "sql_synthesizer-0.2.2-py3-none-any.whl",
    "digest": {
      "sha256": "abc123..."
    }
  }],
  "predicateType": "https://slsa.dev/provenance/v0.2",
  "predicate": {
    "builder": {
      "id": "https://github.com/Actions/github-actions"
    },
    "buildType": "https://github.com/Actions/github-actions@v1",
    "invocation": {
      "configSource": {
        "uri": "git+https://github.com/yourorg/sql-synthesizer@refs/tags/v0.2.2",
        "digest": {
          "sha1": "def456..."
        }
      }
    }
  }
}
```

### 3. Verification Tools

#### CLI Verification
```bash
# Install SLSA verifier
curl -Lo slsa-verifier https://github.com/slsa-framework/slsa-verifier/releases/latest/download/slsa-verifier-linux-amd64
chmod +x slsa-verifier

# Verify package authenticity
./slsa-verifier verify-artifact \
  --provenance-path sql_synthesizer.intoto.jsonl \
  --source-uri github.com/yourorg/sql-synthesizer \
  sql_synthesizer-0.2.2-py3-none-any.whl
```

#### Python Integration
```python
# sql_synthesizer/verification.py
import json
import hashlib
from pathlib import Path

def verify_slsa_provenance(artifact_path: str, provenance_path: str) -> bool:
    """Verify SLSA provenance for distributed artifacts."""
    
    # Calculate artifact digest
    with open(artifact_path, 'rb') as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Load and verify provenance
    with open(provenance_path) as f:
        provenance = json.load(f)
    
    # Verify subject matches
    subjects = provenance.get('subject', [])
    for subject in subjects:
        if subject.get('digest', {}).get('sha256') == artifact_hash:
            return True
    
    return False
```

### 4. Build Hardening

#### Secure Build Environment
```yaml
# Enhanced security for GitHub Actions
jobs:
  secure-build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
      packages: write
    
    steps:
    - uses: actions/checkout@v4
      with:
        persist-credentials: false
    
    - name: Setup Python with hash verification
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        check-latest: true
    
    - name: Install dependencies with hash verification
      run: |
        pip install --require-hashes -r requirements-lock.txt
    
    - name: Build in isolated environment
      run: |
        python -m build --wheel --sdist
    
    - name: Generate SBOM
      uses: anchore/sbom-action@v0
      with:
        path: ./
        format: spdx-json
```

### 5. Dependency Tracking

#### Software Bill of Materials (SBOM)
```yaml
# Generate comprehensive SBOM
- name: Generate SBOM
  uses: anchore/sbom-action@v0
  with:
    path: ./
    format: spdx-json
    output-file: sql-synthesizer.spdx.json

- name: Upload SBOM
  uses: actions/upload-artifact@v3
  with:
    name: sbom
    path: sql-synthesizer.spdx.json
```

#### Requirements Lock File
```bash
# Generate locked requirements with hashes
pip-compile --generate-hashes --output-file requirements-lock.txt requirements.txt
pip-compile --generate-hashes --output-file requirements-dev-lock.txt requirements-dev.txt
```

### 6. Release Process Hardening

#### Signed Releases
```yaml
# Enhanced release workflow with signing
- name: Sign artifacts
  uses: sigstore/gh-action-sigstore-python@v2.1.1
  with:
    inputs: |
      ./dist/*.tar.gz
      ./dist/*.whl
    upload-signing-artifacts: true

- name: Create signed release
  uses: softprops/action-gh-release@v1
  with:
    files: |
      dist/*
      *.sig
      *.crt
    generate_release_notes: true
```

### 7. Consumer Verification

#### Package Installation with Verification
```bash
# For consumers using the package
pip install sql-synthesizer

# Verify SLSA provenance (when available)
pip install slsa-verifier
slsa-verifier verify-artifact \
  --provenance-path provenance.intoto.jsonl \
  --source-uri github.com/yourorg/sql-synthesizer \
  $(pip show -f sql-synthesizer | grep Location)/sql_synthesizer*.whl
```

## Monitoring and Compliance

### 1. Compliance Dashboard
- Build provenance generation success rate
- Artifact signing coverage
- SBOM generation completeness
- Verification tool adoption metrics

### 2. Automated Compliance Checks
```python
# Compliance monitoring script
def check_slsa_compliance():
    """Automated SLSA compliance verification."""
    
    checks = {
        'provenance_exists': verify_provenance_files(),
        'signatures_valid': verify_artifact_signatures(),
        'sbom_complete': verify_sbom_completeness(),
        'build_reproducible': verify_build_reproducibility()
    }
    
    compliance_score = sum(checks.values()) / len(checks)
    return compliance_score >= 0.8  # 80% compliance threshold
```

### 3. Supply Chain Risk Assessment
- Dependency vulnerability scanning
- License compliance verification  
- Maintainer security posture review
- Build environment integrity checks

## Migration Path

### Phase 1: Foundation (Current → SLSA L2)
- ✅ Implement basic provenance generation
- ✅ Set up secure build environment
- ✅ Generate initial SBOM

### Phase 2: Hardening (SLSA L2 → L3)
- 🔄 Implement two-person review enforcement
- 🔄 Add build isolation and hardening
- 🔄 Enhance provenance non-falsifiability

### Phase 3: Advanced (SLSA L3+)
- 🔄 Full hermetic builds
- 🔄 Advanced threat modeling
- 🔄 Supply chain security monitoring

## Resources

- [SLSA Framework Documentation](https://slsa.dev/)
- [GitHub Actions SLSA Generator](https://github.com/slsa-framework/slsa-github-generator)
- [SLSA Verifier Tool](https://github.com/slsa-framework/slsa-verifier)
- [Sigstore Integration Guide](https://docs.sigstore.dev/)

## Manual Setup Required

1. Enable GitHub Advanced Security
2. Configure OpenID Connect for GitHub Actions
3. Set up Sigstore keyless signing
4. Configure supply chain security alerts
5. Implement policy enforcement for SLSA requirements

See [../SETUP_REQUIRED.md](SETUP_REQUIRED.md) for detailed implementation steps.