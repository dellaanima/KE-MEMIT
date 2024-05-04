#edited by me repr_tools 

"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook


def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )


def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    # 모든 템플릿에 정확히 하나의 {} 가 있는지 확인
    # Context Template example : '<s> Therefore, I am writing to urge you. {}', '<s> Because the two characters are both in their . {}', ... 
    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)
    
    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix

            #For llama 
            if 'llama' not in tok.__class__.__name__.lower():
                words[i] = f" {words[i].strip()}"

    # Llama 모델을 사용하는 경우 suffixes에서 시작하는 띄어쓰기 제거
    if 'llama' in tok.__class__.__name__.lower():
        for i in range(len(suffixes)):
            if suffixes[i].startswith(" "):
                suffixes[i] = suffixes[i][1:]  # 첫 번째 띄어쓰기 제거

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    batch_tok = tok([*prefixes, *words, *suffixes],  add_special_tokens=False)
    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]

    last_token_indices = compute_last_token_indices(n, prefixes_len, words_len, suffixes_len, subtoken)
    print('_________________________________________________________________')
    print(last_token_indices)



    if 'llama' in tok.__class__.__name__.lower():
        # Compute indices of last tokens
        if subtoken == "last" or subtoken == "first_after_last":
            return [
                [
                    prefixes_len[i]
                    + words_len[i] + 1 # 앞에 add special token false 로 인해서 ,원래 tokenzie 했을 때보다 <s> 1개 덜 생성되었기 때문  
                    - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
                ]
                # If suffix is empty, there is no "first token after the last".
                # So, just return the last token of the word.
                for i in range(n)
            ]
        elif subtoken == "first":
            return [[prefixes_len[i] + 1 ] for i in range(n)] # 앞에 add special token false 로 인해서 ,원래 tokenzie 했을 때보다 <s> 1개 덜 생성되었기 때문  
        else:
            raise ValueError(f"Unknown subtoken type: {subtoken}") 
        
    else : 
        # Compute indices of last tokens
        if subtoken == "last" or subtoken == "first_after_last":
            return [
                [
                    prefixes_len[i]
                    + words_len[i]
                    - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
                ]
                # If suffix is empty, there is no "first token after the last".
                # So, just return the last token of the word.
                for i in range(n)
            ]
        elif subtoken == "first":
            return [[prefixes_len[i]] for i in range(n)]
        else:
            raise ValueError(f"Unknown subtoken type: {subtoken}")




def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for batch_contexts, batch_idxs in _batch(n=128):
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]


def compute_last_token_indices(n, prefixes_len, words_len, suffixes_len, subtoken):

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            prefixes_len[i]
            + words_len[i]
            - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            for i in range(n)
        ]