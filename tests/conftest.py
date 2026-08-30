"""Temporary CI tracing for persistent KV-cache hang diagnosis."""


def _persistent(nodeid):
    return "test_persistent_gpt_kv_cache" in nodeid


def pytest_runtest_logstart(nodeid, location):
    if _persistent(nodeid):
        print(f"\nPERSISTENT-START {nodeid}", flush=True)


def pytest_runtest_logfinish(nodeid, location):
    if _persistent(nodeid):
        print(f"\nPERSISTENT-FINISH {nodeid}", flush=True)
