from neuroforge.agent.rag import RAGIndex, cite, get_index


def test_index_loads_corpus():
    idx = get_index()
    assert len(idx.doc_ids) >= 5
    assert all(idx.titles)


def test_query_retrieves_relevant_doc():
    cites = cite("TNF-alpha neuroinflammation cytokine", k=2)
    assert cites
    assert cites[0]["doc_id"] == "neuroinflammation"
    assert cites[0]["score"] > 0
    assert "snippet" in cites[0]


def test_dopamine_query():
    cites = cite("dopamine D2 receptor parkinsonian", k=1)
    assert cites and cites[0]["doc_id"] == "dopaminergic"


def test_empty_query_returns_nothing():
    assert RAGIndex().query("", k=2) == []
