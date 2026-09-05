import types

import pytest

from vocab_gen import generate as gen
from vocab_gen.generate import DEFAULT_MODEL, FALLBACK_MODEL, GenerationError, generate
from vocab_gen.providers import classify


def fake_outcome(spec):
    return gen.Outcome(
        types.SimpleNamespace(definition=["d"], candidates=[]),
        types.SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
        spec,
        "low",
    )


@pytest.fixture
def patched(monkeypatch):
    calls = []

    def once(spec, *a, **kw):
        calls.append(spec)
        if spec == gen.resolve_model(DEFAULT_MODEL):
            raise GenerationError("balance", "dashscope", "qwen3.8-flash", "out of credit")
        return fake_outcome(spec)

    monkeypatch.setattr(gen, "_generate_once", once)
    monkeypatch.setattr(gen, "available", lambda _p: True)
    return calls


def test_falls_back_when_the_primary_cannot_answer(patched):
    out = generate("obdurate", ["zephyr"])
    assert out.model == gen.resolve_model(FALLBACK_MODEL)
    assert len(patched) == 2


def test_the_failure_is_carried_not_swallowed(patched):
    out = generate("obdurate", ["zephyr"])
    assert out.fell_back_from is not None
    assert out.fell_back_from.kind == "balance"
    assert "credit" in out.fell_back_from.message


def test_an_explicit_model_is_an_instruction_not_a_hint(patched):
    """--model means run that model; silently using another would be wrong."""
    with pytest.raises(GenerationError):
        generate("obdurate", ["zephyr"], model=DEFAULT_MODEL)
    assert len(patched) == 1


def test_fallback_can_be_switched_off(patched):
    with pytest.raises(GenerationError):
        generate("obdurate", ["zephyr"], allow_fallback=False)


def test_no_fallback_for_a_missing_model(monkeypatch):
    """A typo will not be fixed by asking a different vendor."""
    calls = []

    def once(spec, *a, **kw):
        calls.append(spec)
        raise GenerationError("not_found", "dashscope", "typo", "no such model")

    monkeypatch.setattr(gen, "_generate_once", once)
    monkeypatch.setattr(gen, "available", lambda _p: True)
    with pytest.raises(GenerationError):
        generate("obdurate", ["zephyr"])
    assert len(calls) == 1


def test_no_fallback_when_the_fallback_provider_is_unconfigured(monkeypatch):
    calls = []

    def once(spec, *a, **kw):
        calls.append(spec)
        raise GenerationError("balance", "dashscope", "q", "out of credit")

    monkeypatch.setattr(gen, "_generate_once", once)
    monkeypatch.setattr(gen, "available", lambda _p: False)
    with pytest.raises(GenerationError):
        generate("obdurate", ["zephyr"])
    assert len(calls) == 1


def test_success_reports_no_fallback(monkeypatch):
    monkeypatch.setattr(gen, "_generate_once", lambda spec, *a, **kw: fake_outcome(spec))
    assert generate("obdurate", ["zephyr"]).fell_back_from is None


# --- failure classification --------------------------------------------------


def make(status=None, name="APIStatusError", body=""):
    exc = type(name, (Exception,), {})(body)
    if status is not None:
        exc.status_code = status
    return exc


@pytest.mark.parametrize(
    "exc, kind",
    [
        (make(401, body="invalid x-api-key"), "auth"),
        (make(403), "auth"),
        (make(402, body="payment required"), "balance"),
        (make(429, body="Insufficient balance or no resource package."), "balance"),
        (make(429, body="Too many requests"), "rate_limit"),
        (make(404, body="model not found"), "not_found"),
        (make(name="APIConnectionError", body="failed to connect"), "connection"),
        (make(500, body="server error"), "other"),
    ],
)
def test_classify(exc, kind):
    assert classify(exc) == kind


def test_a_billing_429_is_not_mistaken_for_throughput():
    """These are opposite actions: top up, versus wait and retry."""
    assert classify(make(429, body="please recharge")) == "balance"
    assert classify(make(429, body="rate limit exceeded")) == "rate_limit"


def test_generation_error_serialises_for_a_ui():
    exc = GenerationError("auth", "zhipu", "glm-5.3", "key rejected")
    assert exc.as_dict() == {
        "kind": "auth", "provider": "zhipu", "model": "glm-5.3",
        "message": "key rejected", "also": "", "title": "zhipu",
    }


def test_a_missing_package_is_not_reported_as_a_missing_model():
    """ModuleNotFoundError contains 'NotFound'; substring matching misread it."""
    assert classify(ModuleNotFoundError("No module named 'openai'")) == "setup"
    assert classify(ImportError("cannot import name X")) == "setup"


def test_setup_failure_explains_how_to_fix_the_install():
    from vocab_gen.generate import _explain

    msg = _explain("dashscope", "qwen3.8-flash", ModuleNotFoundError("No module named 'openai'"))
    assert "uv tool install" in msg
    assert "no model" not in msg.lower()


def test_when_both_fail_the_message_names_both(monkeypatch):
    """Reporting only the fallback's error points at the wrong provider."""
    def once(spec, *a, **kw):
        if spec == gen.resolve_model(DEFAULT_MODEL):
            raise GenerationError("balance", "dashscope", "q", "Out of credit.")
        raise GenerationError("auth", "moonshot", "k", "Key rejected.")

    monkeypatch.setattr(gen, "_generate_once", once)
    monkeypatch.setattr(gen, "available", lambda _p: True)
    with pytest.raises(GenerationError) as exc:
        generate("obdurate", ["zephyr"])
    message = exc.value.message
    assert "dashscope" in message and "Out of credit." in message
    assert "moonshot" in message and "Key rejected." in message


def test_a_double_failure_names_both_providers_in_the_title(monkeypatch):
    """Titling it with the fallback alone points at the wrong provider."""
    def once(spec, *a, **kw):
        if spec == gen.resolve_model(DEFAULT_MODEL):
            raise GenerationError("balance", "dashscope", "q", "Out of credit.")
        raise GenerationError("auth", "moonshot", "k", "Key rejected.")

    monkeypatch.setattr(gen, "_generate_once", once)
    monkeypatch.setattr(gen, "available", lambda _p: True)
    with pytest.raises(GenerationError) as exc:
        generate("obdurate", ["zephyr"])
    assert exc.value.title == "dashscope + moonshot"
    assert exc.value.as_dict()["also"] == "dashscope"


def test_a_single_failure_titles_with_one_provider():
    exc = GenerationError("auth", "zhipu", "glm", "nope")
    assert exc.title == "zhipu"
    assert exc.as_dict()["also"] == ""


def test_a_content_filter_rejection_is_retried(monkeypatch):
    """Measured at ~50% on one prompt variant; unretried it biases the sample."""
    from vocab_gen.generate import _with_retries

    monkeypatch.setattr("vocab_gen.generate.time.sleep", lambda _s: None)
    attempts = []

    class Filtered(Exception):
        status_code = 400

    def call():
        attempts.append(1)
        if len(attempts) < 2:
            raise Filtered(
                "Error code: 400 - InternalError.Algo.DataInspectionFailed: "
                "Input text data may contain inappropriate content."
            )
        return "ok"

    assert _with_retries("dashscope", "qwen", call) == "ok"
    assert len(attempts) == 2


def test_an_ordinary_bad_request_is_not_retried(monkeypatch):
    from vocab_gen.generate import _with_retries

    monkeypatch.setattr("vocab_gen.generate.time.sleep", lambda _s: None)
    attempts = []

    class Bad(Exception):
        status_code = 400

    def call():
        attempts.append(1)
        raise Bad("max_tokens must be positive")

    with pytest.raises(Bad):
        _with_retries("dashscope", "qwen", call)
    assert len(attempts) == 1


def test_content_filter_gets_a_smaller_retry_budget(monkeypatch):
    """A classifier scoring a fixed prompt will not change its mind; retrying it
    as hard as a rate limit just spends quota on a failure that cannot recover."""
    from vocab_gen.generate import CONTENT_FILTER_RETRIES, RATE_LIMIT_RETRIES, _with_retries

    assert CONTENT_FILTER_RETRIES < RATE_LIMIT_RETRIES
    monkeypatch.setattr("vocab_gen.generate.time.sleep", lambda _s: None)
    attempts = []

    class Filtered(Exception):
        status_code = 400

    def call():
        attempts.append(1)
        raise Filtered("DataInspectionFailed: Input text data may contain inappropriate content.")

    with pytest.raises(Filtered):
        _with_retries("dashscope", "qwen", call)
    assert len(attempts) == CONTENT_FILTER_RETRIES


def test_an_exhausted_quota_is_billing_not_a_bad_key():
    """Alibaba returns 403 for an exhausted free tier; 'check your key' is wrong."""
    exc = make(403, name="PermissionDeniedError",
               body="The free quota has been exhausted. To continue accessing the model "
                    "on a paid basis, please complete your payment information")
    assert classify(exc) == "balance"
