import json


results = {"name": "py_test", "failed": 0, "skipped": 0, "passed": 0}


def pytest_runtest_logreport(report):
    if report.when == "call":
        if report.passed:
            results["passed"] += 1
        elif report.failed:
            results["failed"] += 1
        elif report.skipped:
            results["skipped"] += 1


def pytest_sessionfinish(session, exitstatus):
    with open("pytest_results.json", "w") as f:
        f.write(f"TEST_CASE={json.dumps(results)}")
