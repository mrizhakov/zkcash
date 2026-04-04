import os
import unittest
import contextlib
from io import StringIO
from pathlib import Path

from tf_bionetta.logging.logger import create_logger
from tf_bionetta.logging.verbose import VerboseMode
from tf_bionetta.logging.pretty import print_success_message


def iterate_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iterate_tests(item)
        else:
            yield item


def main():
    logger = create_logger(VerboseMode.DEBUG)

    failed_tests = []  # List to collect failed test names
    test_modules = []  # Specify which tests to run
    
    root_dir = 'tests'
    root_dir_path = os.path.abspath(root_dir)

    for curr_dir in os.listdir(root_dir_path):
        if curr_dir == '__pycache__' or not os.path.isdir(root_dir_path / Path(curr_dir)):
            continue

        inner_dir = root_dir_path / Path(curr_dir)
        for filename in os.listdir(inner_dir):
            if not filename.endswith('.py') or filename == '__init__.py':
                continue

            module_name = filename[:-3]
            full_module = f"{root_dir}.{curr_dir}.{module_name}"
            test_modules.append(full_module)

    logger.info("Tests to run: " + str(test_modules))


    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")

    for module in test_modules:
        suite.addTests(loader.loadTestsFromName(module))

    os.makedirs("test_logs", exist_ok=True)

    # Custom runner to capture output per test
    print()
    for test in iterate_tests(suite):
        test_name = test.id()
        logger.info("Currently running: " + test_name)

        # Capture output to buffer
        buffer = StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            test_result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(test)

        # If test failed or errored, save output
        if not test_result.failures and not test_result.errors:
            print_success_message("Successfully passed: " + test_name)
            continue
        
        logger.error(test_name)
        for failed_case, tb in test_result.failures + test_result.errors:
            test_name = failed_case.id()
            failed_tests.append(test_name)

            # Clean file name: replace dots with underscores etc.
            safe_name = test_name.replace(".", "_") + ".txt"
            log_file_path = os.path.join("test_logs", safe_name)

            # Save full buffer + traceback to file
            with open(log_file_path, "w") as f:
                f.write(buffer.getvalue())
                f.write("\n")
                f.write(tb)

    # Report
    if failed_tests:
        print("\n❌ Some tests failed:")
        for name in failed_tests:
            print(f" - {name}")
    else:
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()