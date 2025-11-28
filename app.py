import streamlit as st
import time
from core.pipeline import (
    discover_working_models,
    generate_all_responses,
    conduct_peer_evaluation,
    synthesize_final_essay
)

st.set_page_config(page_title="AI Peer Synthesis → Essay", layout="wide")
st.title("🧠 Multi-AI Peer Synthesis → 455-Word Essay")
st.markdown(
    "Uses **all Groq models** for generation & evaluation. "
    "Final synthesis done **only by `llama-3.3-70b-versatile`**."
)

with st.expander("How it works"):
    st.markdown("""
    1. Test all models.
    2. Generate 20 paraphrases.
    3. All models answer all paraphrases.
    4. Cross-model peer evaluation.
    5. Rank & deduplicate top answers.
    6. **Synthesize into 455-word essay using `llama-3.3-70b-versatile` only**.
    """)

api_key = st.text_input("🔑 Groq API Key", type="password")
question = st.text_area("❓ Your Question", height=100)

if st.button("🚀 Generate Essay"):
    if not api_key.strip():
        st.error("Enter API key")
    elif not question.strip():
        st.error("Enter a question")
    else:
        start = time.time()
        try:
            with st.status("🔍 Testing models...") as s:
                models = discover_working_models(api_key)
                if not models:
                    st.error("No working models")
                    st.stop()
                s.update(label=f"✅ Working: {', '.join(models)}", state="complete")

            with st.status("🤖 Generating responses..."):
                responses = generate_all_responses(api_key, question, models)
                valid = [r for r in responses if r["content"] != "[No valid response]"]
                st.write(f"✅ {len(valid)} valid responses")

            with st.status("📊 Peer evaluating..."):
                evals = conduct_peer_evaluation(api_key, responses, models)
                st.write(f"✅ {len(evals)} evaluations")

            with st.status("🏆 Ranking top answers..."):
                resp_map = {r['id']: r for r in responses}
                enriched = []
                for ev in evals:
                    resp = resp_map.get(ev['target_response_id'])
                    if resp and resp['content'] != "[No valid response]":
                        ev['full_answer'] = resp['content']
                        enriched.append(ev)
                enriched.sort(key=lambda x: x['score'], reverse=True)
                seen, top = set(), []
                for ev in enriched:
                    key = ev['full_answer'][:80].lower()
                    if key not in seen:
                        seen.add(key)
                        top.append(ev)
                    if len(top) >= 15:
                        break
                st.write(f"✅ {len(top)} top answers selected")

            with st.status("✍️ Synthesizing final essay..."):
                essay = synthesize_final_essay(api_key, top)
                elapsed = time.time() - start

            st.subheader("📄 Final Essay (455 words)")
            st.text_area("", essay, height=500)
            st.success(f"Done in {elapsed:.1f}s | Words: {len(essay.split())}")

        except Exception as e:
            st.exception(e)
