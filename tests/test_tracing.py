from caregraph.obs import InMemoryTracer, NoopTracer


def test_inmemory_tracer_records_spans_and_attrs() -> None:
    tracer = InMemoryTracer()
    with tracer.span("outer", stage="a") as s:
        s.set("n", 3)
    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.name == "outer"
    assert span.attributes == {"stage": "a", "n": 3}
    assert span.duration_ms >= 0


def test_noop_tracer_is_silent() -> None:
    tracer = NoopTracer()
    with tracer.span("x") as s:
        s.set("k", 1)  # must not raise
    assert not hasattr(tracer, "spans")


def test_reset_clears_spans() -> None:
    tracer = InMemoryTracer()
    with tracer.span("a"):
        pass
    tracer.reset()
    assert tracer.spans == []
