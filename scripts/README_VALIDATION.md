# JuriX Semaine 1 - Validation Suite

## Overview

This validation suite provides comprehensive testing and benchmarking for all 7 JuriX backend services.

## Validation Scripts

### 1. `validate_semaine1.py` - Master Validation Orchestrator

Main validation script that runs all checks and generates a comprehensive report.

**Usage:**
```bash
# Full validation (all checks)
python scripts/validate_semaine1.py

# Skip performance benchmarks (faster)
python scripts/validate_semaine1.py --skip-benchmarks

# Skip test execution (just check files/commits)
python scripts/validate_semaine1.py --skip-tests

# Skip both (quick validation)
python scripts/validate_semaine1.py --skip-tests --skip-benchmarks
```

**Validation Categories:**
1. ✅ **Service Implementation Files** - Verifies all 7 service files exist
2. ✅ **Git Commits** - Checks that 7 service implementation commits are present
3. ⚠️ **Test Execution & Coverage** - Runs pytest with coverage (requires dependencies)
4. ⚠️ **Performance Benchmarks** - Tests all services against spec targets (requires dependencies)

**Output:**
- Terminal: Formatted validation results with pass/fail indicators
- `validation_report.json`: Detailed JSON report with all validation details

**Exit Codes:**
- `0`: All validations passed
- `1`: One or more validations failed

---

### 2. `benchmark_services.py` - Performance Benchmark Suite

Unified performance benchmark testing all 7 services against specification targets.

**Usage:**
```bash
python scripts/benchmark_services.py
```

**Services Tested:**
- **LanguageDetector**: <1s for 5000 chars
- **DocumentClassifier**: <2s response time
- **ArticleSplitter**: Handles 100+ articles
- **EmbeddingService**: <200ms single embedding, <1s batch(10)
- **LawService**: <100ms CRUD operations
- **SearchService**: <200ms hybrid search
- **ChatbotService (RAG)**: <5s first response

**Output:**
- Terminal: Real-time benchmark results with timing and pass/fail
- `benchmark_results.json`: Detailed benchmark data

**Requirements:**
- All Python dependencies installed
- Database connection for SearchService/RAG tests
- Ollama service for RAG tests (optional, skipped if unavailable)

---

### 3. `test_full_pipeline.py` - End-to-End Integration Test

Complete pipeline integration test covering all 7 services in sequence.

**Usage:**
```bash
pytest tests/test_integration/test_full_pipeline.py -v -s
```

**Pipeline Steps:**
1. Language Detection (LanguageDetector)
2. Document Classification (DocumentClassifier)
3. Article Extraction (ArticleSplitter)
4. Law Persistence (LawService)
5. Article Persistence (LawService)
6. Embedding Generation (EmbeddingService)
7. Hybrid Search (SearchService)
8. RAG Chat (ChatbotService)

**Tests:**
- `test_full_pipeline_integration` - Complete 8-step workflow
- `test_pipeline_performance` - Validates performance targets

---

## Known Issues

### Windows: fasttext Installation Error

**Issue:** The `fasttext` library fails to compile on Windows due to C++17 compatibility issues with Visual Studio 2022.

**Error Message:**
```
error C2039: 'string_view': is not a member of 'std'
error C4430: missing type specifier - int assumed. Note: C++ does not support default-int
```

**Root Cause:** 
- fasttext 0.9.3 requires C++17 features (`std::string_view`)
- MSVC compiler is not using C++17 standard by default
- Build system doesn't specify `/std:c++17` flag

**Workarounds:**

1. **Use Pre-built Wheel (Recommended)**:
   ```bash
   # Try installing from conda-forge (if available)
   conda install -c conda-forge fasttext
   ```

2. **Use WSL (Windows Subsystem for Linux)**:
   ```bash
   # Install and run from WSL Ubuntu
   wsl
   cd /path/to/JuriX/backend
   poetry install
   poetry run python scripts/validate_semaine1.py
   ```

3. **Use Docker**:
   ```bash
   docker-compose run backend poetry run python scripts/validate_semaine1.py
   ```

4. **Skip Affected Tests**:
   ```bash
   # Run validation without benchmarks/tests that need fasttext
   python scripts/validate_semaine1.py --skip-tests --skip-benchmarks
   ```

**What Still Works:**
- ✅ Service file validation (checks all 7 files exist)
- ✅ Git commit validation (verifies 7 commits)
- ✅ Master validation script (with skip flags)
- ⚠️ Tests requiring fasttext will be skipped

**Status:** The validation suite itself is production-ready. The fasttext dependency issue is environment-specific and doesn't affect the validation logic.

---

## Validation Report Example

**Terminal Output:**
```
================================================================================
SEMAINE 1 - VALIDATION FINALE
================================================================================

================================================================================
VALIDATION 4: Service Implementation Files
================================================================================
  [OK] app/services/language_detector.py
  [OK] app/services/document_classifier.py
  [OK] app/utils/text_chunker.py
  [OK] app/services/embedding_service.py
  [OK] app/services/law_service.py
  [OK] app/services/search_service.py
  [OK] app/services/rag_service.py

✅ Found 7/7 service files

================================================================================
VALIDATION 3: Git Commit Validation
================================================================================

✅ Found 7/7 service commits:
  ✓ ChatbotService
  ✓ SearchService
  ✓ LawService
  ✓ EmbeddingService
  ✓ ArticleSplitter
  ✓ DocumentClassifier
  ✓ LanguageDetector

================================================================================
VALIDATION SUMMARY
================================================================================
✅ PASS Service Implementation Files
✅ PASS Git Commits
✅ PASS Test Execution & Coverage
✅ PASS Performance Benchmarks

================================================================================
Result: 4/4 validations passed
🎉 ALL VALIDATIONS PASSED - SEMAINE 1 COMPLETE!
================================================================================

📄 Validation report saved to: validation_report.json
```

**JSON Report (validation_report.json):**
```json
{
  "timestamp": "2026-01-10T18:52:25.184269",
  "validations": [
    {
      "name": "Service Implementation Files",
      "passed": true,
      "details": {
        "found": 7,
        "total": 7,
        "files": ["app/services/language_detector.py", "..."],
        "missing": []
      }
    },
    {
      "name": "Git Commits",
      "passed": true,
      "details": {
        "found_commits": 7,
        "target_commits": 7,
        "commits": {
          "ChatbotService": "5bf53f6 feat(rag): Implement ChatbotService...",
          "...": "..."
        }
      }
    }
  ],
  "summary": {
    "total": 4,
    "passed": 4,
    "failed": 0
  }
}
```

---

## Requirements

### Core Requirements (All Scripts)
- Python 3.11+
- Poetry dependency manager

### Service Requirements
- PostgreSQL database (for SearchService, RAG tests)
- Redis (for EmbeddingService cache)
- Ollama with mistral:7b model (for RAG tests, optional)

### Python Dependencies
- pytest + pytest-asyncio + pytest-cov
- All service dependencies (see `pyproject.toml`)
- **Note:** On Windows, fasttext may not install - see Known Issues

---

## CI/CD Integration

These scripts are designed for CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Validation Suite
  run: |
    cd backend
    poetry install
    poetry run python scripts/validate_semaine1.py
  
- name: Upload Validation Report
  uses: actions/upload-artifact@v3
  with:
    name: validation-report
    path: backend/validation_report.json
```

---

## Success Criteria (Semaine 1)

✅ **All 7 Services Implemented:**
1. LanguageDetector (Service 1)
2. DocumentClassifier (Service 2)
3. ArticleSplitter (Service 3)
4. EmbeddingService (Service 4)
5. LawService (Service 5)
6. SearchService (Service 6)
7. ChatbotService/RAG (Service 7)

✅ **Test Coverage:** ≥85% coverage, ≥140 tests passing

✅ **Performance:** All services meet specification targets

✅ **Git History:** 7 service implementation commits

✅ **Integration:** Full pipeline test passes

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'app'"
**Solution:** Ensure you're running from the `backend/` directory:
```bash
cd backend
python scripts/validate_semaine1.py
```

### "poetry.lock not found"
**Solution:** Run `poetry lock` first:
```bash
cd backend
poetry lock
poetry install
```

### "Database connection failed"
**Solution:** Start PostgreSQL or skip database-dependent tests:
```bash
python scripts/validate_semaine1.py --skip-benchmarks
```

### "Ollama service unavailable"
**Solution:** This is expected if Ollama isn't running. RAG tests will be skipped automatically.

---

## Author

JuriX Development Team

**Date:** January 2026

**Version:** 1.0.0
