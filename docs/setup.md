# Setup

## PDM for Dependency Management

We use [pdm](https://pdm-project.org/en/latest/) for cross-platform dependency management. 

On Linux, here's a simple way to install: 
```
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
echo "
```

For detailed instructions or troubleshooting, see [official docs](https://pdm-project.org/en/latest/#installation). 

## Pre-commit hooks

Ensure pre-commit hooks are installed
``` bash
pdm run pre-commit install
```