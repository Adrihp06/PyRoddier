import unittest
import coverage
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_tests():
    # Initialize coverage
    cov = coverage.Coverage(
        branch=True,
        source=['src'],
        omit=['*/tests/*', '*/__pycache__/*']
    )
    cov.start()

    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    # Stop coverage and generate report
    cov.stop()
    cov.save()

    # Print coverage report
    print("\nCoverage Report:")
    cov.report()

    # Generate HTML report
    cov.html_report(directory='coverage_report')
    print("\nHTML coverage report generated in 'coverage_report' directory")

    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())