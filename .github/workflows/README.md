## GitHub Actions Workflows  
This project includes two automated workflows:

###  **Continuous Integration (ci.yml)**
**Location:** `.github/workflows/ci.yml`  
📌 **What it does:**  
- Runs on every push or pull request.  
- Checks out the latest code.  
- Sets up Python 3.10.  
- Installs dependencies from `requirements.txt`.  
- Runs automated tests using `pytest`.  
- Ensures the codebase remains stable.

### **Code Linting (lint.yml)**
**Location:** `.github/workflows/lint.yml`  
📌 **What it does:**  

- Runs on every push or pull request.
- Checks Python code formatting using `flake8`.
- Enforces coding standards (PEP8).
- Helps maintain a clean and readable codebase.
