import agent_ia_langgraph


def test_package_import() -> None:
    assert callable(agent_ia_langgraph.main)
