from .handler import SinglePromptHandler


def make_ioi_single(model):
    return SinglePromptHandler(
        model=model,
        clean_prompt="When John and Mary went to the shops, John gave a bag to",
        corrupt_prompt=" When Alice and Bob went to the shops, Charlie gave a bag to",
        answer=" Mary",
        wrong_answer=" John",
    )
