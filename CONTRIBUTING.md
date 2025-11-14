# Contributing to Plant Health Classification

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/plant-health-classification.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt`

## Project Structure

```
plant-health-classification/
├── docs/                  # Documentation files
├── models/               # Model implementations
├── utils/                # Utility functions
├── train.py             # Training script
├── evaluate.py          # Evaluation script
└── example.py           # Example usage
```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### Documentation

- Update relevant documentation when making changes
- Add comments for complex logic
- Update README.md if adding new features
- Maintain the documentation in `docs/` directory

### Testing

Before submitting:
1. Test your changes thoroughly
2. Ensure existing functionality still works
3. Verify code syntax: `python -m py_compile your_file.py`

### Commit Messages

Write clear, descriptive commit messages:
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 72 characters
- Add detailed description if needed

Example:
```
Add attention visualization for ViT model

- Implement attention map extraction
- Add visualization utilities
- Update documentation
```

## Types of Contributions

### Bug Fixes

- Check if the bug is already reported in Issues
- Create a new issue if not exists
- Submit a PR with the fix

### New Features

- Discuss major features in Issues first
- Ensure features align with project goals
- Update documentation and tests

### Documentation

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve existing documentation

### Code Optimization

- Profile code before optimizing
- Benchmark improvements
- Document performance gains

## Areas for Contribution

### High Priority

1. **Dataset Utilities**
   - Scripts to download PlantVillage dataset
   - Data augmentation experiments
   - Dataset statistics visualization

2. **Model Improvements**
   - Hybrid architectures (CNN + Transformer)
   - Model compression techniques
   - Transfer learning implementations

3. **Evaluation Tools**
   - Attention visualization for ViT
   - Grad-CAM for interpretability
   - Error analysis tools

4. **Training Enhancements**
   - Learning rate finder
   - Mixed precision training
   - Distributed training support

### Medium Priority

5. **Deployment**
   - ONNX export
   - Model quantization
   - Web API (FastAPI/Flask)
   - Mobile deployment (TorchScript)

6. **Experiments**
   - Different ViT configurations
   - Ensemble methods
   - Self-supervised pre-training

7. **Documentation**
   - Jupyter notebooks with examples
   - Video tutorials
   - Blog posts

### Low Priority

8. **Testing**
   - Unit tests for models
   - Integration tests
   - Continuous Integration setup

9. **Tools**
   - Docker containerization
   - Makefile for common tasks
   - Configuration management

## Pull Request Process

1. Update documentation for your changes
2. Test your changes thoroughly
3. Update CHANGELOG.md if applicable
4. Submit PR with clear description
5. Link related issues
6. Wait for review and address feedback

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code optimization

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

## Code Review

All submissions require review. We will:
- Check code quality and style
- Test functionality
- Review documentation
- Provide constructive feedback

## Questions?

- Open an issue for questions
- Tag with "question" label
- We'll respond as soon as possible

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Acknowledgments

Thank you for contributing to advancing plant health detection technology!
