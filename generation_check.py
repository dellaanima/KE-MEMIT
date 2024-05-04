result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    expect=knowledge["attribute"],
                    kind=kind,
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                )

# knowledge["prompt"], 가  prompt 로 사용됨. 그러면 prompt 의 형태는?  subject 가 들어간, prompt 
# expect=knowledge["attribute"], 
'''
[
  {
    "known_id": 0,
    "subject": "Vinson Massif",
    "attribute": "Antarctica",
    "template": "{} is located in the continent",
    "prediction": " of Antarctica. It is the largest of the three",
    "prompt": "Vinson Massif is located in the continent of",
    "relation_id": "P30"
  },
  {
    "known_id": 1,
    "subject": "Beats Music",
    "attribute": "Apple",
    "template": "{} is owned by",
    "prediction": " Apple, which is also the owner of Beats Electronics",
    "prompt": "Beats Music is owned by",
    "relation_id": "P127"
  },
'''

def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    if expect is not None and answer.strip() != expect:
        
        return dict(correct_prediction=False)
    


def predict_token(model, tokenizer, prompts, return_p=False):
    inp = make_inputs(tokenizer, prompts)
    preds, p = predict_from_input(model, inp)
    result = [tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result

def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p

# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )
