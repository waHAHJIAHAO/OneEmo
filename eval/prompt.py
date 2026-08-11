ERG_EVAL_PROMPT = """You are an experienced and impartial expert in psychology.

**Task Definition:** You are evaluating a single model response in an Empathetic Response Generation (ERG) task. The model is replying to a user who has just expressed a personal feeling or situation. Your evaluation must focus solely on the quality of the response content. **Do NOT reward longer responses, more elaborate formatting, or any specific writing style. A short, genuine response can score higher than a long, generic one.**

**Context :**
{{context}}

**Chat history:**
{{chat_history}}

**Seeker Response:**
`{{seeker_utterance}}`

**Model Response:**
`{{model_response}}`

Evaluate the response across the following three dimensions using an integer Likert scale of **1 (Poor) to 5 (Excellent)**.

---

### **Dimension 1: Empathy**
Assess whether the response accurately recognizes the seeker’s emotional state and offers an emotionally appropriate reaction (e.g., validation, warmth, understanding). Penalize responses that misidentify the emotion, sound robotic, or offer unsolicited advice that ignores the feeling.

| Score | Description |
|-------|-------------|
| 1 | Completely misses or contradicts the seeker’s emotional state; response is cold, dismissive, or overtly inappropriate. |
| 2 | Shows minimal recognition of emotion; response is generic (e.g., "I’m sorry to hear that") with no evidence of true understanding. |
| 3 | Acknowledges the emotion at a surface level; response is adequate but lacks depth, specificity, or genuine warmth. |
| 4 | Demonstrates clear understanding of the emotional state; response is warm, validating, and reasonably tailored to the seeker’s situation. |
| 5 | Deeply and precisely captures the nuanced emotional state; response feels genuinely human, highly validating, and emotionally resonant. |

**Score:** `{{1-5}}`

---

### **Dimension 2: Coherence & Consistency**
Assess whether the response is logically consistent with the seeker’s context and the conversational flow. Check for contradictions, irrelevant topic shifts, or factual/logical errors. Length and style must not influence this score.

| Score | Description |
|-------|-------------|
| 1 | Completely incoherent, contradictory, or irrelevant to the seeker’s message. |
| 2 | Mostly inconsistent or contains noticeable logical gaps; relevance is weak. |
| 3 | Generally coherent and relevant, but contains minor inconsistencies, vague connections, or slight drift. |
| 4 | Consistent and well-connected to the context; logical flow is clear with only trivial issues. |
| 5 | Perfectly coherent, logically airtight, and seamlessly aligned with the seeker’s context and intent. |

**Score:** `{{1-5}}`

---

### **Dimension 3: Informativeness**
Assess the substantive value of the response beyond mere emotional acknowledgment. Does it provide meaningful perspective, useful information, a relevant question, or a non-templatized insight? **Penalize empty platitudes, repetitive template phrases, or generic filler** (e.g., "Everything will be fine," "Stay strong," "I understand"). A concise but substantive response should score higher than a verbose but hollow one.

| Score | Description |
|-------|-------------|
| 1 | Purely templatized or meaningless filler; adds zero informational or conversational value. |
| 2 | Mostly generic platitudes with negligible substantive content; could apply to any situation. |
| 3 | Contains some meaningful content but still relies noticeably on generic or templatized language. |
| 4 | Substantive and reasonably specific; provides genuine value (insight, relevant question, useful framing) with minimal templatization. |
| 5 | Highly informative and uniquely tailored; offers profound insight, a precisely targeted question, or genuinely useful perspective with no detectable templatization. |

**Score:** `{{1-5}}`

---

**Final Output Format (strictly follow this):**
Empathy: score(integer1-5)
Coherence: score(integer1-5)
Informativeness: score(integer1-5)
"""


ESC_EVAL_PROMPT = """You are an experienced and impartial expert in psychology.
**Task Definition:** You are evaluating a single model turn in an Emotional Support Conversation (ESC). The seeker is in distress and looking for psychological support. The model’s role is to provide professional-quality emotional support. **Score based on support quality only. Do NOT reward longer responses, more elaborate formatting, or any specific writing style. A brief, skillful response can score higher than a lengthy, undirected one.**

**Chat history::**
`{{chat_history}}`

**patient:**
`{{patient_utterance}}`

**Model Response:**
`{{model_response}}`

Evaluate the response across the following three dimensions using an integer Likert scale of **1 (Poor) to 5 (Excellent)**.

---

### **Dimension 1: Empathy**
Assess whether the response accurately recognizes the patient’s emotional state and makes them feel understood. Focus only on emotional attunement, validation, warmth, and sensitivity to nuance. Do not score strategy use here unless it directly affects felt empathy.

| Score | Description |
|-------|-------------|
| 1 | Misses, dismisses, or contradicts the patient’s feelings; cold or invalidating. |
| 2 | Minimal acknowledgment of emotion; generic or perfunctory. |
| 3 | Recognizes the general feeling but remains surface-level. |
| 4 | Clearly understands and validates the patient’s emotional state with warmth and presence. |
| 5 | Precisely captures the emotional nuance and feels deeply human, attuned, and validating. |

**Score:** `{{1-5}}`

---

### **Dimension 2: Support Skill**
Assess the appropriate and correct use of evidence-based emotional support strategies, such as: **questioning** (open-ended exploration), **emotional reflection** (paraphrasing feelings), **self-disclosure** (relevant, bounded sharing), **affirmation/reassurance** (realistic, not empty), **information/advice** (when solicated or clearly needed), and **structuring/instilling hope**. Penalize misuse of skills (e.g., premature advice, excessive self-disclosure, interrogative questioning) and reward precise, timely strategy selection. Length is irrelevant.

| Score | Description |
|-------|-------------|
| 1 | No discernible support strategy, or actively harmful technique (e.g., toxic positivity, blame, excessive advice). |
| 2 | Attempts a strategy but executes poorly (e.g., closed yes/no question, irrelevant self-disclosure, shallow reassurance). |
| 3 | Uses one or more recognizable support strategies correctly, but selection or timing is suboptimal (e.g., advice before validation). |
| 4 | Deploys appropriate strategies with good timing and execution; demonstrates competent conversational support skills. |
| 5 | Masterful, context-sensitive use of multiple support strategies with impeccable timing; reflects professional-level support skill. |

**Score:** `{{1-5}}`

---

### **Dimension 3: Overall Effectiveness**
Assess the holistic impact of the response on the seeker, taking into account the conversation history and this round of conversation. If this were the only turn you could judge, how likely is it that the seeker would feel heard, emotionally relieved, and meaningfully supported? This is the aggregate score reflecting the combined success of empathy and skill in producing tangible emotional benefit. **A concise, perfectly timed response should score higher than a rambling, unfocused one.**

| Score | Description |
|-------|-------------|
| 1 | Likely to leave the seeker feeling worse, unheard, or frustrated; response is counter-therapeutic. |
| 2 | Minimal positive impact; seeker would likely feel little change or mild disappointment. |
| 3 | Moderately helpful; seeker receives some comfort or direction, but the response lacks lasting impact or full trust. |
| 4 | Clearly helpful; seeker would likely feel noticeably heard, validated, and somewhat emotionally relieved. |
| 5 | Exceptionally effective; seeker would likely feel deeply understood, genuinely comforted, and measurably better equipped to cope. |

**Score:** `{{1-5}}`

---

**Final Output Format (strictly follow this):**
Empathy:  score(integer1-5)
Skill: score(integer1-5)
Overall: score(integer1-5)
"""
