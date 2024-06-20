from sae_eap.core.types import Model
from sae_eap.data.handler import SinglePromptHandler


def make_single_prompt_handler(model: Model) -> SinglePromptHandler:
    return SinglePromptHandler(
        model=model,
        clean_prompt="hello",
        corrupt_prompt="hello",
        answer=" world",
        wrong_answer=" world",
    )


def test_single_prompt_handler_tensor_shapes(ts_model: Model):
    handler = make_single_prompt_handler(ts_model)
    assert handler.get_batch_size() == 1

    assert len(handler.clean_tokens.shape) == 2
    assert handler.clean_tokens.shape[0] == handler.get_batch_size()
    assert len(handler.answer_tokens.shape) == 1
    assert handler.answer_tokens.shape[0] == handler.get_batch_size()
    assert len(handler.wrong_answer_tokens.shape) == 1
    assert handler.wrong_answer_tokens.shape[0] == handler.get_batch_size()

    # Test the function with a simple example
    logits = handler.get_logits(ts_model, input="clean")
    assert len(logits.shape) == 3
    assert logits.shape[0] == handler.get_batch_size()
    assert logits.shape[1] == handler.get_n_pos()
    assert logits.shape[2] == ts_model.cfg.d_vocab
    metric = handler.get_metric(logits)
    assert len(metric.shape) == 1
    assert metric.shape[0] == handler.get_batch_size()
