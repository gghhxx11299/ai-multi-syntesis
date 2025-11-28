import streamlit as st
import requests
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Dict, Optional

# ----------------------------
# Configuration
# ----------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

CANDIDATE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905",
]

MODEL_TPM_LIMITS = {
    "llama-3.1-8b-instant": 6000,
    "llama-3.3-70b-versatile": 12000,
    "qwen/qwen3-32b": 6000,
    "moonshotai/kimi-k2-instruct-0905": 10000,
}

PARAPHRASE_COUNT = 20
MAX_GLOBAL_RPM = 60

# Rate-limiting state (per session)
global_call_timestamps = []
model_token_usage = {}

# ----------------------------
# Utility Functions
# ----------------------------
def estimate_tokens(text: str) -> int:
    return max(10, int(len(text) / 3.5) + 20)

def wait_for_global_rate_limit():
    now = time.time()
    global global_call_timestamps
    global_call_timestamps[:] = [t for t in global_call_timestamps if now - t < 60]
    if len(global_call_timestamps) >= MAX_GLOBAL_RPM:
        oldest = min(global_call_timestamps)
        sleep_time = 60 - (now - oldest) + 0.2
        if sleep_time > 0:
            time.sleep(sleep_time)
    global_call_timestamps.append(time.time())

def wait_for_model_tpm_limit(model: str, estimated_tokens: int):
    if model not in MODEL_TPM_LIMITS:
        return
    limit = MODEL_TPM_LIMITS[model]
    now = time.time()
    usage = model_token_usage.get(model, [])
    usage[:] = [(t, tok) for t, tok in usage if now - t < 60]
    current_usage = sum(tok for _, tok in usage)
    if current_usage + estimated_tokens > limit:
        if usage:
            oldest_time = min(t for t, _ in usage)
            sleep_time = 60 - (now - oldest_time) + 1.0
        else:
            sleep_time = 1.0
        if sleep_time > 0:
            time.sleep(sleep_time)
    usage.append((now, estimated_tokens))
    model_token_usage[model] = usage

def call_groq(api_key: str, model: str, messages: List[Dict], max_tokens: int = 800, retries: int = 2) -> Optional[str]:
    prompt_text = " ".join(msg.get("content", "") for msg in messages)
    estimated_input_tokens = estimate_tokens(prompt_text)
    total_estimated = estimated_input_tokens + max_tokens

    wait_for_model_tpm_limit(model, total_estimated)
    wait_for_global_rate_limit()

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    data = {"model": model, "messages": messages, "max_tokens": max_tokens}

    try:
        response = requests.post(GROQ_URL, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
            return content.strip() if content else None
        elif response.status_code == 429 and retries > 0:
            time.sleep(8)
            return call_groq(api_key, model, messages, max_tokens, retries - 1)
    except Exception:
        pass
    return None

def generate_fallback_paraphrases(question: str) -> List[str]:
    templates = [
        f"What are the key dimensions of: {question}",
        f"How might different experts interpret: {question}",
        f"What are the underlying assumptions in: {question}",
        f"Explore {question} from historical, cultural, and ethical angles",
        f"What would a contrarian say about: {question}",
        f"How does {question} manifest in real-world contexts?",
        f"What are the unresolved tensions in: {question}",
        f"Reframe {question} as a design challenge",
        f"What metaphors help explain: {question}",
        f"How would you teach someone about: {question}",
    ]
    base = templates[:10]
    for i in range(10, 20):
        base.append(f"Perspective {i+1}: {question}")
    return base

def extract_score_insight_value(eval_text: str):
    if not eval_text:
        return 5.0, "No insight.", "No value."
    score = 5.0
    score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', eval_text, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
    insight = "No insight extracted."
    insight_match = re.search(r'INSIGHT:\s*(.+?)(?:\n|$)', eval_text, re.IGNORECASE)
    if insight_match:
        insight = insight_match.group(1).strip()
    value = "No value explanation."
    value_match = re.search(r'VALUE:\s*(.+?)(?:\n|$)', eval_text, re.IGNORECASE)
    if value_match:
        value = value_match.group(1).strip()
    return score, insight, value

# ----------------------------
# Pipeline Functions
# ----------------------------
def discover_working_models(api_key: str) -> List[str]:
    working = []
    test_prompt = "Respond with only: OK"
    for model in CANDIDATE_MODELS:
        resp = call_groq(api_key, model, [{"role": "user", "content": test_prompt}], max_tokens=5)
        if resp and "ok" in resp.lower():
            working.append(model)
    return working

def generate_20_paraphrases(api_key: str, question: str, working_models: List[str]) -> List[str]:
    if not working_models:
        return generate_fallback_paraphrases(question)
    model = working_models[0]
    prompt = f"""Generate exactly 20 distinct paraphrased versions of this question. Each must preserve the core intent but vary in framing, emphasis, angle, cultural perspective, or interpretive lens.

Original: "{question}"

Return ONLY a numbered list (1–20), one per line, no extra text."""
    messages = [{"role": "user", "content": prompt}]
    response = call_groq(api_key, model, messages, max_tokens=600)
    if not response:
        return generate_fallback_paraphrases(question)
    paraphrases = []
    for line in response.split('\n'):
        cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', line.strip())
        if cleaned and len(cleaned) > 20 and not cleaned.startswith(('Original', 'Note', 'Return', 'Generate', 'Answer')):
            paraphrases.append(cleaned)
    while len(paraphrases) < 20:
        paraphrases.append(f"Analyze comprehensively: {question} (variation {len(paraphrases)+1})")
    return paraphrases[:20]

def _generate_single_response(api_key: str, model: str, paraphrase_id: int, paraphrase: str) -> Dict:
    prompt = f"Answer this question with depth, precision, and originality:\n\n\"{paraphrase}\""
    messages = [{"role": "user", "content": prompt}]
    response = call_groq(api_key, model, messages, max_tokens=600)
    content = response.strip() if response and len(response.strip()) > 50 else "[No valid response]"
    return {
        'id': f"{model}_p{paraphrase_id}",
        'model': model,
        'paraphrase_id': paraphrase_id,
        'paraphrase': paraphrase,
        'content': content
    }

def generate_all_responses(api_key: str, question: str, working_models: List[str]) -> List[Dict]:
    paraphrases = generate_20_paraphrases(api_key, question, working_models)
    all_responses = []
    max_workers = min(8, len(working_models) * len(paraphrases))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_generate_single_response, api_key, model, i, para)
            for model in working_models
            for i, para in enumerate(paraphrases)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_responses.append(result)
    return all_responses

def _evaluate_single_response(api_key: str, evaluator: str, target_resp: Dict, paraphrase_id: int) -> Dict:
    eval_prompt = f"""You are critically evaluating this answer to the question:
"{target_resp['paraphrase']}"

Answer:
"{target_resp['content'][:1000]}..."

Do THREE things:
1. Rate it 1–10 on insight, accuracy, depth, originality, clarity.
2. Extract the SINGLE MOST VALUABLE INSIGHT in one clear sentence.
3. Explain why that insight matters in one sentence.

Respond EXACTLY in this format:
SCORE: X/10
INSIGHT: [one sentence]
VALUE: [one sentence]"""
    messages = [{"role": "user", "content": eval_prompt}]
    eval_text = call_groq(api_key, evaluator, messages, max_tokens=250)
    if not eval_text:
        eval_text = "SCORE: 5/10\nINSIGHT: Evaluation failed.\nVALUE: No value could be determined."
    score, insight, value = extract_score_insight_value(eval_text)
    return {
        'evaluator': evaluator,
        'target_model': target_resp['model'],
        'paraphrase_id': paraphrase_id,
        'score': score,
        'insight': insight,
        'value': value,
        'feedback': eval_text,
        'target_response_id': target_resp['id']
    }

def conduct_peer_evaluation(api_key: str, all_responses: List[Dict], working_models: List[str]) -> List[Dict]:
    grouped = defaultdict(list)
    for r in all_responses:
        grouped[r['paraphrase_id']].append(r)
    evaluations = []
    with ThreadPoolExecutor(max_workers=min(12, 100)) as executor:
        futures = []
        for p_id, responses in grouped.items():
            for resp in responses:
                evaluators = [m for m in working_models if m != resp['model']] or working_models
                for evaluator in evaluators:
                    futures.append(
                        executor.submit(_evaluate_single_response, api_key, evaluator, resp, p_id)
                    )
        for future in as_completed(futures):
            result = future.result()
            if result:
                evaluations.append(result)
    return evaluations

def synthesize_final_essay(api_key: str, top_answers: List[Dict]) -> str:
    synthesis_model = "llama-3.3-70b-versatile"  # ONLY this model for final synthesis
    compilation = "\n\n---\n".join([
        f"High-Scoring Insight: {a['insight']}\nAnswer: {a['full_answer']}"
        for a in top_answers
    ])
    synthesis_prompt = f"""You are a world-class synthesizer. Create a single, cohesive, insightful essay of exactly 455 words from the following top-rated AI answers.

Requirements:
- Formal, academic tone.
- Unified narrative: introduction, development, conclusion.
- Integrate the deepest insights; eliminate redundancy.
- Do NOT mention models, scores, or evaluation.
- The final output must be precisely 455 words.

Inputs:
{compilation}

Now write the 455-word essay:"""
    messages = [{"role": "user", "content": synthesis_prompt}]
    essay = call_groq(api_key, synthesis_model, messages, max_tokens=2000)
    return essay if essay else "Final synthesis failed."

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="AI Peer Synthesis → 455-Word Essay", layout="wide")
st.title("🧠 Multi-AI Peer Synthesis → Final Essay (455 words)")
st.markdown(
    "Uses **all Groq models** for generation and peer evaluation. "
    "Final synthesis into a **455-word essay** is performed **exclusively by `llama-3.3-70b-versatile`**."
)

with st.expander("How it works"):
    st.markdown("""
    1. Tests all 4 models for availability.
    2. Generates 20 paraphrased versions of your question.
    3. Each model answers all 20 versions (parallel).
    4. Models peer-evaluate each other’s answers.
    5. Top answers are ranked and deduplicated.
    6. **Only `llama-3.3-70b-versatile` synthesizes the final 455-word essay**.
    """)

api_key = st.text_input("🔑 Enter your Groq API Key", type="password")
question = st.text_area("❓ Enter your question", height=100)

if st.button("🚀 Generate 455-Word Essay"):
    if not api_key.strip():
        st.error("Please enter your Groq API key.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        start_time = time.time()
        try:
            with st.status("🔍 Testing models...") as status:
                working_models = discover_working_models(api_key)
                if not working_models:
                    st.error("No models are working. Check your API key and internet connection.")
                    st.stop()
                status.update(label=f"✅ Working models: {', '.join(working_models)}", state="complete")

            with st.status("🤖 Generating responses..."):
                all_responses = generate_all_responses(api_key, question, working_models)
                valid_responses = [r for r in all_responses if r["content"] != "[No valid response]"]
                st.write(f"✅ Generated {len(valid_responses)} valid responses.")

            with st.status("📊 Conducting peer evaluations..."):
                evaluations = conduct_peer_evaluation(api_key, all_responses, working_models)
                st.write(f"✅ Completed {len(evaluations)} evaluations.")

            with st.status("🏆 Ranking top answers..."):
                resp_map = {r['id']: r for r in all_responses}
                enriched = []
                for ev in evaluations:
                    resp = resp_map.get(ev['target_response_id'])
                    if resp and resp['content'] != "[No valid response]":
                        ev['full_answer'] = resp['content']
                        enriched.append(ev)
                enriched.sort(key=lambda x: x['score'], reverse=True)
                seen = set()
                top_list = []
                for ev in enriched:
                    key = ev['full_answer'][:80].lower()
                    if key not in seen:
                        seen.add(key)
                        top_list.append(ev)
                    if len(top_list) >= 15:
                        break
                st.write(f"✅ Selected {len(top_list)} top unique answers.")

            with st.status("✍️ Synthesizing final essay using llama-3.3-70b-versatile..."):
                final_essay = synthesize_final_essay(api_key, top_list)
                processing_time = time.time() - start_time

            st.subheader("📄 Final Synthesized Essay (455 words)")
            st.text_area("Essay", final_essay, height=500)
            word_count = len(final_essay.split())
            st.success(f"✅ Completed in {processing_time:.1f} seconds | Word count: {word_count}")

        except Exception as e:
            st.exception(f"An error occurred: {e}")
